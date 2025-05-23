import torch
from torch import nn


class LinkPredictor(nn.Module):
    """
    MLP für Drug–Protein Link Prediction mit unterschiedlicher Input-Dimension.
    """

    def __init__(
        self,
        in_drug: int,   # Embedding-Dimension der Drugs
        in_protein: int,   # Embedding-Dimension der Proteine
        hidden_dim: int    # gemeinsame Hidden-/Output-Dimension
    ):
        super().__init__()
        # 1) Projektion der unterschiedlichen Embedding-Dimensionen in einen gemeinsamen Raum
        self.src_proj = nn.Linear(in_drug, hidden_dim, bias=False)
        self.dst_proj = nn.Linear(in_protein, hidden_dim, bias=False)

        # 2) MLP auf dem zusammengeführten Vektor
        self.lin1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        drug_emb: torch.Tensor,      # [N_src, in_dim_src]
        protein_emb: torch.Tensor,   # [N_dst, in_dim_dst]
        edge_index: torch.LongTensor # [2, E]  row0=drug idx, row1=protein idx
    ) -> torch.Tensor:
        src_idx, dst_idx = edge_index

        # 1) Hole die Rohembeddings für die Kanten-Paare
        h_src_raw = drug_emb[src_idx]      # [E, in_dim_src]
        h_dst_raw = protein_emb[dst_idx]   # [E, in_dim_dst]

        # 2) Projiziere in den gemeinsamen Hidden-Raum
        h_src = self.src_proj(h_src_raw)   # [E, hidden_dim]
        h_dst = self.dst_proj(h_dst_raw)   # [E, hidden_dim]

        # 3) Konkatenation
        h = torch.cat([h_src, h_dst], dim=1)  # [E, 2*hidden_dim]

        # 4) MLP
        x = self.lin1(h)   # [E, hidden_dim]
        x = self.relu(x)
        x = self.lin2(x)   # [E, 1]

        # 5) Rückgabe roher Logits als [E]
        return x.view(-1)