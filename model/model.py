import time
from typing import Optional

import torch
from torch import nn, Tensor
from torch_geometric.data import HeteroData
from tqdm import tqdm

from model.attention import DrugProteinAttention
from model.conv import HypergraphConv, HypergraphAttantion
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
            hidden: int = 10,
            heads: int = 4,
            dropout: float = 0.1,
            num_rounds: int = 3,  # ← how many mp rounds
    ):
        super().__init__()
        self.num_rounds = num_rounds
        self.drug_hyper = HypergraphAttantion(drug_in,
                                              hidden)
        self.protein_hyper = HypergraphConv(protein_in,
                                            hidden)

        self.drug_back_projection = nn.Linear(hidden, drug_in, bias=False)

        self.protein_back_projection = nn.Linear(hidden, protein_in, bias=False)

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

    def forward(self, data: HeteroData,
                pos_edge_index: Tensor,
                neg_edge_index: Optional[Tensor] = None,
                num_rounds: Optional[int] = None) -> Tensor:
        # unpack features & topology
        x_drug = data['drug_atom'].x
        edge_drug = data['drugs_hyperedge'].x
        inc_drug = data['drug_atom', 'to', 'drug_hyperedge'].edge_index

        x_prot = data['proteins'].x
        prot_inc = data['protein_amino', 'to'].edge_index

        if neg_edge_index is not None:
            edge_index_all = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        else:
            edge_index_all = pos_edge_index

        # multi‐round message passing
        num_rounds = num_rounds if num_rounds is not None else self.num_rounds
        #for i in tqdm(range(num_rounds), desc='Rounds'):
        start_time = time.time()

        x_drug = self.drug_hyper(x_drug, inc_drug, edge_drug)
        #tqdm.write(f"Drug hypergraph conv: {time.time() - start_time:.3f}s")

        drug_time = time.time()
        x_prot = self.protein_hyper(x_prot, prot_inc)
        #tqdm.write(f"Protein hypergraph conv: {time.time() - drug_time:.3f}s")

        # cross‐type attention updates protein (and optionally drugs)
        # returns updated (drug, protein)
        att_time = time.time()
        x_drug, x_prot = self.drug_protein_att(x_drug, x_prot, edge_index_all)
        #tqdm.write(f"Cross attention: {time.time() - att_time:.3f}s")

        proj_time = time.time()
        x_drug = self.drug_back_projection(x_drug)
        x_prot = self.protein_back_projection(x_prot)
        #tqdm.write(f"Back projection: {time.time() - proj_time:.3f}s")

        #tqdm.write(f"Total round time: {time.time() - start_time:.3f}s")

        # final link‐prediction on drug→protein edges
        pred_time = time.time()
        logits = self.link_pred(x_drug, x_prot, edge_index_all)
        #tqdm.write(f"Link prediction: {time.time() - pred_time:.3f}s")
        return logits
