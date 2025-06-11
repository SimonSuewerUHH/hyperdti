import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax

class HypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        concat: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat = True
        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:

        num_nodes = x.size(0)
        num_edges = int(hyperedge_index[1].max()) + 1

        hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = 1, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, 1, 1) * out

        return out


class HypergraphAttantion(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            attention_mode: str = 'node',
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention_mode = attention_mode

        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')
        self.lin_edge = Linear(10, heads * out_channels, bias=False,
                          weight_initializer='glorot')
        self.att = Parameter(torch.empty(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self,
                x: Tensor,
                hyperedge_index: Tensor,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:

        num_nodes = x.size(0)
        num_edges = int(hyperedge_index[1].max()) + 1


        hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)


        hyperedge_attr = self.lin_edge(hyperedge_attr)
        hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                             self.out_channels)


        x_i = x[hyperedge_index[0]]
        x_j = hyperedge_attr[hyperedge_index[1]]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if self.attention_mode == 'node':
            alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
        else:
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


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
        start = time.time()
        Wh = self.lin(x)  # [N, D]
        We = self.edge_lin(edge_attr) if self.has_edge_attr else None  # [E, D]
        print(
            f"[CustomHyperSemanticMessagePassing] Linear layers took {time.time() - start:.4f} seconds")

        # 1) Alle (e,u)-Paare sammeln
        start = time.time()
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
        print(
            f"[CustomHyperSemanticMessagePassing] Collecting pairs took {time.time() - start:.4f} seconds")

        # 2) Key/Value Tensor erstellen
        start = time.time()
        K_list = []
        V_list = []
        for e, u in pairs:
            k = Wh[u] + (We[e] if We is not None else 0)
            K_list.append(k)
            V_list.append(Wh[u])
        K_all = torch.stack(K_list, dim=0)  # [P, D]
        V_all = torch.stack(V_list, dim=0)  # [P, D]
        print(
            f"[CustomHyperSemanticMessagePassing] Creating key/value tensors took {time.time() - start:.4f} seconds")

        # 3) Pro Zielknoten v batchen
        start = time.time()
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
        print(
            f"[CustomHyperSemanticMessagePassing] Batching took {time.time() - start:.4f} seconds")

        # 4) Attention im Batch
        start = time.time()
        Q = Wh.unsqueeze(1)  # [N,1,D]
        attn_out, _ = self.multihead_attn(
            query=Q,
            key=K_padded,
            value=V_padded,
            key_padding_mask=mask
        )
        out = attn_out.squeeze(1)  # [N,D]
        print(
            f"[CustomHyperSemanticMessagePassing] Attention took {time.time() - start:.4f} seconds")
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

    def forward(self, x: torch.Tensor, edge_x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        # HypergraphConv expects: x (num_nodes, in_channels), incidence (num_hyperedges, num_nodes)
        import time
        start1 = time.time()
        h = self.conv1(x, edge_index, edge_x)
        conv1_time = time.time() - start1
        print(f"[DrugHyperConv] Conv1 took {conv1_time:.4f} seconds")

        start2 = time.time()
        h = self.relu(h)
        h = self.dropout(h)
        relu1_time = time.time() - start2
        print(f"[DrugHyperConv]  Relu1 + dropout took {relu1_time:.4f} seconds")

        start3 = time.time()
        h = self.conv2(h, edge_index, edge_x)
        conv2_time = time.time() - start3
        print(f"[DrugHyperConv]  Conv2 took {conv2_time:.4f} seconds")

        start4 = time.time()
        h = self.relu(h)
        relu2_time = time.time() - start4
        print(f"[DrugHyperConv]  Relu2 took {relu2_time:.4f} seconds")
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
