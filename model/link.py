import torch
from torch import nn, Tensor


class LinkPredictor(nn.Module):
    def __init__(self, in_drug: int, in_prot: int, hidden: int, dropout: float = 0.5):
        super().__init__()
        self.proj_src = nn.Linear(in_drug, hidden, bias=False)
        self.proj_dst = nn.Linear(in_prot, hidden, bias=False)
        self.gate = nn.Linear(2 * hidden, 2 * hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.bilinear = nn.Bilinear(hidden, hidden, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.proj_src.reset_parameters()
        self.proj_dst.reset_parameters()
        self.gate.reset_parameters()
        for m in self.mlp:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.bilinear.reset_parameters()

    def forward(self, x_drug: Tensor, x_prot: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        h_src = self.proj_src(x_drug[src])
        h_dst = self.proj_dst(x_prot[dst])
        h_cat = torch.cat([h_src, h_dst], dim=1)
        gate = torch.sigmoid(self.gate(h_cat))
        h = gate * h_cat
        mlp_score = self.mlp(h).view(-1)
        bil_score = self.bilinear(h_src, h_dst).view(-1)
        return mlp_score + bil_score
