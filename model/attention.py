import torch
from torch import nn
import torch.nn.functional as F

class DrugProteinAttention(nn.Module):
    """
    Vektorisierte, bidirektionale Multi-Head-Attention zwischen Drug- und Protein-Knoten.
    """
    def __init__(
        self,
        in_drug: int,
        in_protein: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # lineare Projektionen
        self.src_lin = nn.Linear(in_drug, out_dim, bias=False)
        self.dst_lin = nn.Linear(in_protein, out_dim, bias=False)
        # batched MultiheadAttention (batch_first=True)
        self.attn_src2dst = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.attn_dst2src = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        # Norm & Dropout für Residual-Pfade
        self.norm_src = nn.LayerNorm(out_dim)
        self.norm_dst = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x_src: torch.Tensor,            # [N_src, in_drug]
        x_dst: torch.Tensor,            # [N_dst, in_protein]
        edge_index: torch.LongTensor    # [2, E]
    ) -> (torch.Tensor, torch.Tensor):
        h_src = self.src_lin(x_src)      # [N_src, out_dim]
        h_dst = self.dst_lin(x_dst)      # [N_dst, out_dim]
        N_src, N_dst = h_src.size(0), h_dst.size(0)

        # ---- Drug → Protein ----
        # Queries für alle Proteine in einem Batch
        Q_dst = h_dst.unsqueeze(1)       # [N_dst, 1, out_dim]
        # Keys & Values: repeat h_src für jedes Protein
        K_src = h_src.unsqueeze(0).expand(N_dst, -1, -1)  # [N_dst, N_src, out_dim]
        V_src = K_src.clone()
        # Maske: True an Positionen, die NICHT attendiert werden sollen
        mask_src2dst = torch.ones((N_dst, N_src), dtype=torch.bool, device=h_src.device)
        src_idx, dst_idx = edge_index  # beide Länge E
        mask_src2dst[dst_idx, src_idx] = False

        attn_dst, _ = self.attn_src2dst(
            Q_dst, K_src, V_src,
            key_padding_mask=mask_src2dst
        )  # [N_dst, 1, out_dim]
        updated_dst = attn_dst.squeeze(1)  # [N_dst, out_dim]

        # Residual + Norm + Aktivation
        updated_dst = self.norm_dst(h_dst + self.drop(updated_dst))
        updated_dst = F.relu(updated_dst)

        # ---- Protein → Drug ----
        Q_src = h_src.unsqueeze(1)       # [N_src, 1, out_dim]
        K_dst = h_dst.unsqueeze(0).expand(N_src, -1, -1)  # [N_src, N_dst, out_dim]
        V_dst = K_dst.clone()
        mask_dst2src = torch.ones((N_src, N_dst), dtype=torch.bool, device=h_src.device)
        mask_dst2src[src_idx, dst_idx] = False

        attn_src, _ = self.attn_dst2src(
            Q_src, K_dst, V_dst,
            key_padding_mask=mask_dst2src
        )  # [N_src, 1, out_dim]
        updated_src = attn_src.squeeze(1)  # [N_src, out_dim]

        updated_src = self.norm_src(h_src + self.drop(updated_src))
        updated_src = F.relu(updated_src)

        return updated_src, updated_dst