import torch
from torch import nn
import torch.nn.functional as F

class DrugProteinAttention(nn.Module):
    """
    Module for multi-head attention message passing from Drug to Protein nodes.
    """
    def __init__(
        self,
        in_drug: int,
        in_protein: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Projektionen in denselben Embed­dingspace
        self.src_lin = nn.Linear(in_drug, out_dim, bias=False)
        self.dst_lin = nn.Linear(in_protein, out_dim, bias=False)
        # Zwei Attention-Module: Src→Dst und Dst→Src
        self.attn_src2dst = nn.MultiheadAttention(embed_dim=out_dim,
                                                  num_heads=num_heads,
                                                  batch_first=True)
        self.attn_dst2src = nn.MultiheadAttention(embed_dim=out_dim,
                                                  num_heads=num_heads,
                                                  batch_first=True)

    def forward(self,
                x_src: torch.Tensor,
                x_dst: torch.Tensor,
                edge_index: torch.LongTensor
                ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x_src (Tensor[N_src, in_dim_src]): Embeddings der Quellknoten (Drugs)
            x_dst (Tensor[N_dst, in_dim_dst]): Embeddings der Zielknoten (Proteine)
            edge_index (LongTensor[E, 2]):     Jede Zeile [src_id, dst_id]
        Returns:
            updated_src (Tensor[N_src, out_dim]),
            updated_dst (Tensor[N_dst, out_dim])
        """
        N_src, N_dst = x_src.size(0), x_dst.size(0)

        # 1) Projektion
        h_src = self.src_lin(x_src)  # [N_src, out_dim]
        h_dst = self.dst_lin(x_dst)  # [N_dst, out_dim]

        # 2) Adjazenz-Listen aufbauen
        node_to_src = [[] for _ in range(N_dst)]
        node_to_dst = [[] for _ in range(N_src)]
        for s, d in edge_index.T.tolist():
            node_to_src[d].append(s)
            node_to_dst[s].append(d)

        # 3) Attention: Src → Dst
        updated_dst = torch.zeros_like(h_dst)
        for v in range(N_dst):
            src_ids = node_to_src[v]
            if not src_ids:
                updated_dst[v] = h_dst[v]
                continue
            # Keys & Values aus Drugs
            K = h_src[src_ids]  # [L, out_dim]
            V = h_src[src_ids]  # [L, out_dim]
            # Query aus Protein
            Q = h_dst[v].unsqueeze(0).unsqueeze(0)  # [1,1,out_dim]
            attn_out, _ = self.attn_src2dst(Q, K.unsqueeze(0), V.unsqueeze(0))
            updated_dst[v] = attn_out.view(-1)

        # 4) Attention: Dst → Src
        updated_src = torch.zeros_like(h_src)
        for u in range(N_src):
            dst_ids = node_to_dst[u]
            if not dst_ids:
                updated_src[u] = h_src[u]
                continue
            # Keys & Values aus Proteinen
            K = h_dst[dst_ids]  # [L, out_dim]
            V = h_dst[dst_ids]  # [L, out_dim]
            # Query aus Drug
            Q = h_src[u].unsqueeze(0).unsqueeze(0)  # [1,1,out_dim]
            attn_out, _ = self.attn_dst2src(Q, K.unsqueeze(0), V.unsqueeze(0))
            updated_src[u] = attn_out.view(-1)

        # 5) Nichtlinearität
        return F.relu(updated_src), F.relu(updated_dst)
