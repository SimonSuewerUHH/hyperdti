import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


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

    def forward(self, x, incidence, edge_attr):
        """
        x:         [N, in_dim]
        incidence: [E, N]  (Hypergraph-Inzidenzmatrix)
        edge_attr: [E, edge_dim]
        """
        N = x.size(0)
        Wh = self.lin(x)                             # [N, D]
        We = self.edge_lin(edge_attr) if self.has_edge_attr else None  # [E, D]

        # 1) Alle (e,u)-Paare sammeln
        pairs = []
        pair_owner = []  # für jedes Paar, zu welchem Zielknoten v es gehört
        for v in range(N):
            # Kanten, die v enthalten
            e_ids = torch.nonzero(incidence[:, v], as_tuple=False).flatten()
            for e in e_ids:
                u_ids = torch.nonzero(incidence[e], as_tuple=False).flatten()
                for u in u_ids:
                    pairs.append((e.item(), u.item()))
                    pair_owner.append(v)
        if not pairs:
            return F.relu(Wh)

        # 2) Key/Value Tensor erstellen
        K_list = []
        V_list = []
        for e, u in pairs:
            k = Wh[u] + (We[e] if We is not None else 0)
            K_list.append(k)
            V_list.append(Wh[u])
        K_all = torch.stack(K_list, dim=0)  # [P, D]
        V_all = torch.stack(V_list, dim=0)  # [P, D]

        # 3) Pro Zielknoten v batchen
        P = len(pairs)
        owners = torch.tensor(pair_owner, dtype=torch.long, device=x.device)  # [P]
        # split K_all/V_all nach owners
        grouped_K = []
        grouped_V = []
        masks = []
        for v in range(N):
            idx = (owners == v).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                # Leer-Sequence für v
                grouped_K.append(torch.zeros((0, Wh.size(1)), device=x.device))
                grouped_V.append(torch.zeros((0, Wh.size(1)), device=x.device))
            else:
                grouped_K.append(K_all[idx])
                grouped_V.append(V_all[idx])
        # Padding auf längste Sequence
        K_padded = pad_sequence(grouped_K, batch_first=True)  # [N, L_max, D]
        V_padded = pad_sequence(grouped_V, batch_first=True)  # [N, L_max, D]
        # Maske: True an Paddings
        mask = torch.arange(K_padded.size(1), device=x.device).unsqueeze(0) \
             >= torch.tensor([g.size(0) for g in grouped_K], device=x.device).unsqueeze(1)
        # 4) Attention im Batch
        Q = Wh.unsqueeze(1)        # [N,1,D]
        attn_out, _ = self.multihead_attn(
            query=Q,
            key=K_padded,
            value=V_padded,
            key_padding_mask=mask
        )
        out = attn_out.squeeze(1)  # [N,D]
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
