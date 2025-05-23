import torch
from torch import nn
import torch.nn.functional as F



class CustomHyperSemanticMessagePassing(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads=8):
        """
        in_dim:    Dimension der Eingabe-Knotenfeatures
        out_dim:   Dimension der Ausgabe-Knotenfeatures
        edge_dim:  Dimension der edge_attr (Kantenfeatures)
        num_heads: Anzahl der Attention-Heads
        """
        super().__init__()
        # 1) Knotentransformation
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        # 2) Projektion der Kantenfeatures auf dieselbe Dim wie die Knotenausgabe
        self.has_edge_attr = edge_dim > 0
        if self.has_edge_attr:
            self.edge_lin = nn.Linear(edge_dim, out_dim, bias=False)
        # 3) Multi-Head-Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x:          Tensor [N, in_dim] mit den Knoteneingabe-Features
        edge_index: LongTensor [2, E], erste Zeile Quelle u, zweite Zeile Ziel v
        edge_attr:  Tensor [E, edge_dim] mit Kantenfeatures (z.B. Gewichtungen)
        """
        N = x.size(0)

        # 1) Knotentransformation
        Wh = self.lin(x)  # [N, out_dim]

        # 2) Edge-Feature-Projektion
        if self.has_edge_attr:
            We = self.edge_lin(edge_attr)  # [E, out_dim]

        # 3) Ergebnis-Container
        out = torch.zeros_like(Wh)

        # 4) Für jeden Knoten v: finde alle Hyperedges, in denen er drin ist
        for v in range(N):
            # Liste aller Edge-IDs, bei denen incidence[e, v] == 1
            e_ids = torch.nonzero(edge_index[:, v], as_tuple=False).flatten().tolist()
            if not e_ids:
                continue

            # 5) Sammle (edge, node)-Paare für Attention
            keys = []
            values = []
            for e_id in e_ids:
                # alle Knoten u in Hyperedge e_id
                u_ids = torch.nonzero(edge_index[e_id], as_tuple=False).flatten().tolist()
                for u in u_ids:
                    # Key = Wh[u] plus Kanten-Bias We[e_id]
                    if self.has_edge_attr:
                        keys.append(Wh[u] + We[e_id])
                    else:
                        keys.append(Wh[u])
                    # Value = Wh[u]
                    values.append(Wh[u])

            # Staple zu Tensoren [L, out_dim]
            key_tensor = torch.stack(keys, dim=0)
            value_tensor = torch.stack(values, dim=0)

            # Bereite Query/Key/Value für MultiheadAttention vor
            # Query: das Feature des Zielknotens v
            query = Wh[v].unsqueeze(0).unsqueeze(0)  # [1, 1, out_dim]
            key = key_tensor.unsqueeze(0)  # [1, L, out_dim]
            value = value_tensor.unsqueeze(0)  # [1, L, out_dim]

            # 6) Multihead-Attention
            attn_out, _ = self.multihead_attn(query, key, value)
            # attn_out: [1,1,out_dim] → speichere in out[v]
            out[v] = attn_out.view(-1)

        # 7) Nichtlinearität & Rückgabe
        return F.relu(out)

class DrugHyperConv(nn.Module):
    """
    Module for hypergraph convolution on drug-atom to drug-hyperedge incidence.
    """
    def __init__(
            self,
            in_dim: int,
            edge_dim: int,
            hidden_dim: int,
            out_dim: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = CustomHyperSemanticMessagePassing(in_dim, hidden_dim, edge_dim)
        self.conv2 = CustomHyperSemanticMessagePassing(hidden_dim, out_dim, edge_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # HypergraphConv expects: x (num_nodes, in_channels), incidence (num_hyperedges, num_nodes)
        h = self.conv1(x, edge_index, edge_x)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_x)
        h = self.relu(h)
        return h


class ProteinHyperConv(nn.Module):
    """
    Module for hypergraph convolution on protein-amino to protein-hyperedge incidence.
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = CustomHyperSemanticMessagePassing(in_dim, hidden_dim, 0)
        self.conv2 = CustomHyperSemanticMessagePassing(hidden_dim, out_dim, 0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index, None)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, None)
        h = self.relu(h)
        return h
