import torch
from torch import nn
from torch_geometric.data import HeteroData

from model.attention import DrugProteinAttention
from model.conv import DrugHyperConv, ProteinHyperConv
from model.link import LinkPredictor


class HeteroHyperModel(nn.Module):
    """
    End-to-end heterogeneous hypergraph model for drug-protein link prediction.
    """

    def __init__(
            self,
            drug_in: int,
            drug_edge: int,
            protein_in: int,
            hidden: int = 64,
            heads: int = 4,
            dropout: float = 0.1,
            num_rounds: int = 3,  # ← how many mp rounds
    ):
        super().__init__()
        self.num_rounds = num_rounds
        self.drug_hyper = nn.ModuleList([
            DrugHyperConv(drug_in,
                          drug_edge,
                          hidden,
                          hidden,
                          dropout)
            for i in range(num_rounds)
        ])
        self.protein_hyper = nn.ModuleList([
            ProteinHyperConv(protein_in,
                             hidden,
                             hidden,
                             dropout)
            for i in range(num_rounds)
        ])

        self.drug_back_projection = nn.ModuleList([
            nn.Linear(hidden, drug_in, bias=False)
            for i in range(num_rounds)
        ])
        self.protein_back_projection = nn.ModuleList([
            nn.Linear(hidden, protein_in, bias=False)
            for i in range(num_rounds)
        ])

        # cross‐type attention (shared across rounds)
        self.drug_protein_att = DrugProteinAttention(
            in_drug=hidden,
            in_protein=hidden,
            out_dim=hidden,
        )

        self.link_pred = LinkPredictor(
            in_drug=drug_in,
            in_protein=protein_in,
            hidden_dim=hidden)

    def forward(self, data: HeteroData) -> torch.Tensor:
        # unpack features & topology
        x_drug = data['drug_atom'].x
        edge_drug = data['drugs_hyperedge'].x
        inc_drug = data['drug_atom', 'to', 'drug_hyperedge'].edge_index

        x_prot = data['proteins'].x
        prot_inc = data['protein_amino', 'to'].edge_index

        dp_edge_idx = data['drug', 'to', 'protein'].edge_index

        # multi‐round message passing
        for i in range(self.num_rounds):
            x_drug = self.drug_hyper[i](x_drug, edge_drug, inc_drug)
            x_prot = self.protein_hyper[i](x_prot,  prot_inc)

            # cross‐type attention updates protein (and optionally drugs)
            # returns updated (drug, protein)
            x_drug, x_prot = self.drug_protein_att(x_drug, x_prot, dp_edge_idx)
            x_drug = self.drug_back_projection[i](x_drug)
            x_prot = self.protein_back_projection[i](x_prot)

        # final link‐prediction on drug→protein edges
        logits = self.link_pred(x_drug, x_prot, dp_edge_idx)
        return logits
