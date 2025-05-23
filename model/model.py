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
    ):
        super().__init__()
        self.drug_hyper = DrugHyperConv(drug_in, drug_edge, hidden, hidden, dropout)
        self.protein_hyper = ProteinHyperConv(protein_in, hidden, hidden, dropout)
        self.drug_protein_att = DrugProteinAttention(
            in_drug=hidden,
            in_protein=hidden,
            out_dim=hidden,
            num_heads=heads,
            dropout=dropout,
        )
        self.link_pred = LinkPredictor(
            in_drug=hidden,
            in_protein=hidden,
            hidden_dim=hidden)

    def forward(self, data: HeteroData) -> torch.Tensor:
        # Drug hypergraph
        drug_atom_x = data['drug_atom'].x
        drug_edge_x = data['drugs_hyperedge'].x
        drug_inc = data['drug_atom', 'to', 'drug_hyperedge'].edge_index
        drug_emb = self.drug_hyper(drug_atom_x, drug_edge_x, drug_inc)

        # Protein hypergraph
        protein_amino_x = data['proteins'].x
        prot_inc = data['protein_amino', 'to'].edge_index
        protein_emb = self.protein_hyper(protein_amino_x, prot_inc)

        # Cross-type attention
        dp_edge_index = data['drug', 'to', 'protein'].edge_index
        # Update protein embeddings by attending over drugs
        drug_emb, protein_emb = self.drug_protein_att(drug_emb, protein_emb, dp_edge_index)

        # Link prediction logits for each edge
        logits = self.link_pred(drug_emb, drug_emb, dp_edge_index)
        return logits
