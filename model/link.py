import torch
from torch import nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    """
    MLP für Drug–Protein Link Prediction mit erweiterten Verbesserungen:
    - Projektion in gemeinsamen Raum
    - Gating-Mechanismus
    - Tiefe MLP mit Residual-Verbindungen und lernbarem Residual-Gewicht
    - BatchNorm + Dropout
    - Kombination aus MLP- und Bilinear-Scoring
    - Explizites Reset der Parameter
    """

    def __init__(
        self,
        in_drug: int,
        in_protein: int,
        hidden_dim: int,
        dropout_p: float = 0.5
    ):
        super().__init__()
        # Projektion der unterschiedlichen Embedding-Dimensionen
        self.src_proj = nn.Linear(in_drug, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(in_protein, hidden_dim, bias=False)

        # Gating auf konkatenierten Embeddings
        self.gate = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        # Tiefe MLP-Layer
        self.lin1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 1)

        # Lernbarer Residual-Gewicht-Vektor
        self.res_weight = nn.Parameter(torch.ones(hidden_dim))

        # Bilinear-Scoring
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

        # Regularisierung
        self.dropout = nn.Dropout(p=dropout_p)

        # Parameter initialisieren
        self.reset_parameters()

    def reset_parameters(self):
        # Reset der Linearen Layer
        self.src_proj.reset_parameters()
        self.dst_proj.reset_parameters()
        self.gate.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        # BatchNorm Layer
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        # Bilinear
        self.bilinear.reset_parameters()
        # Residual-Gewichte auf 1 setzen
        nn.init.ones_(self.res_weight)

    def forward(
        self,
        drug_emb: torch.Tensor,      # [N_src, in_drug]
        protein_emb: torch.Tensor,   # [N_dst, in_protein]
        edge_index: torch.LongTensor # [2, E]
    ) -> torch.Tensor:
        src_idx, dst_idx = edge_index

        # Rohembeddings für Kanten-Paare
        # 1) Hole die Rohembeddings für die Kanten-Paare
        h_src_raw = drug_emb[src_idx]  # [E, in_dim_src]
        h_dst_raw = protein_emb[dst_idx]  # [E, in_dim_dst]

        # 2) Projiziere in den gemeinsamen Hidden-Raum
        h_src = self.src_proj(h_src_raw)  # [E, hidden_dim]
        h_dst = self.dst_proj(h_dst_raw)  # [E, hidden_dim]

        # Konkateniertes Feature
        h_cat = torch.cat([h_src, h_dst], dim=1)  # [E, 2*hidden_dim]

        # Gating
        gate = torch.sigmoid(self.gate(h_cat))    # [E, 2*hidden_dim]
        h = gate * h_cat                          # [E, 2*hidden_dim]

        # Erster MLP-Layer
        x1 = self.lin1(h)                         # [E, hidden_dim]
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        # Zweiter MLP-Layer mit lernbarem Residual-Gewicht
        raw_x2 = self.lin2(x1)                    # [E, hidden_dim]
        # gewichtete Residual-Verbindung: raw_x2 * w + x1 * (1 - w)
        x2 = raw_x2 * self.res_weight + x1 * (1 - self.res_weight)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)

        # MLP-Score
        mlp_score = self.lin3(x2).view(-1)        # [E]

        # Bilinear-Score
        bil_score = self.bilinear(h_src, h_dst).view(-1)  # [E]

        # Kombiniertes Score-Output
        return mlp_score + bil_score