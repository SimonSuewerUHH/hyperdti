from collections import defaultdict
import os

import numpy as np
import torch
from torch_geometric.data import HeteroData

from preprocessing.drug import create_shared_hypergraph_with_labels
from preprocessing.protein import get_model, create_shared_protein_hypergraph


class HypergraphDataGenerator:
    """
    Generiert einen heterogenen Hypergraphen für Drugs basierend auf SMILES.
    Protein-Teil noch nicht implementiert.

    Parameter:
      - drug_smiles_list: Liste von SMILES-Strings für Drugs
      - feature_dim: Dimension der Knotenfeatures
      - seed: Zufallssaat für Reproduzierbarkeit (noch nicht genutzt für RDKit)
    """

    def __init__(self,
                 drug_smiles_list,
                 proteins,
                 sequences):
        self.drug_smiles_list = drug_smiles_list
        self.proteins = proteins
        self.sequences = sequences
        self.proteins_sequences = list(zip(proteins, sequences))
        self.protein_model, self.protein_alphabet = get_model()

    def find_key_by_value(self, combined_node_labels, value):
        for n_id, node_value in combined_node_labels.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if (node_value.startswith(value) and node_value.endswith(value)) or node_value == value:
                return n_id

    def get_hyper_edge_index_drug_to_proteins(self, drug_to_proteins, node_labels, global_node_labels_p, sparse_incidence_matrix=True):
        """
        Gibt den Hypergraphen als Incidence-Matrix zurück.
        """
        if sparse_incidence_matrix:
            raise NotImplementedError("Sparse Incidence Matrix is not implemented yet.")
        else:
            protein_to_drug = []
            for drug, proteins in drug_to_proteins.items():
                drug_node_id = self.find_key_by_value(node_labels, drug)
                protein_ids = [self.find_key_by_value(global_node_labels_p, x) for x in proteins]
                for n in protein_ids:
                    protein_to_drug.append([drug_node_id, n])

            return torch.tensor(protein_to_drug).t().contiguous()

    def get_hyper_edge_index(self, hyperedges, num_nodes,
                                          sparse_incidence_matrix=True):
        """
        Gibt den Hypergraphen als Incidence-Matrix zurück.
        """
        if sparse_incidence_matrix:
            num_hyperedges = len(hyperedges)
            incidence = torch.zeros((num_hyperedges, num_nodes), dtype=torch.long)
            for he_idx, nodes in enumerate(hyperedges.values()):
                incidence[he_idx, nodes] = 1
            return incidence
        else:
            he_index_map = {he: idx for idx, he in enumerate(hyperedges.keys())}
            atom_to_he = []
            for he_label, nodes in hyperedges.items():
                he_idx = he_index_map[he_label]
                for n in nodes:
                    atom_to_he.append([n, he_idx])
            return torch.tensor(atom_to_he).t().contiguous()

    def generate(self) -> HeteroData:
        data = HeteroData()

        (node_labels,
         hyperedges,
         trade_names,
         nodes_features,
         edge_features) = create_shared_hypergraph_with_labels(
            self.drug_smiles_list,
            plot=False,
            plot_individual=False
        )

        data['drug_atom'].x = torch.from_numpy(np.stack(nodes_features)).float()
        data['drugs_hyperedge'].x = torch.from_numpy(np.stack(edge_features)).float()

        num_nodes = len(nodes_features)
        data['drug_atom', 'to', 'drug_hyperedge'].edge_index = self.get_hyper_edge_index(hyperedges,
                                                                                         num_nodes)

        drug_proteins = list(zip(trade_names, self.proteins))
        drug_to_proteins = defaultdict(list)
        for drug, protein in drug_proteins:
            drug_to_proteins[drug].append(protein)

        drug_to_proteins = dict(drug_to_proteins)

        global_node_labels_p, global_hyperedges_p, global_node_features_p = create_shared_protein_hypergraph(
            self.proteins_sequences, self.protein_model, self.protein_alphabet, plot=False)

        data['proteins'].x = torch.from_numpy(np.stack(global_node_features_p)).float()

        num_amino = len(global_node_features_p)
        data['protein_amino', 'to'].edge_index = self.get_hyper_edge_index(global_hyperedges_p,
                                                                           num_amino)

        data['drug', 'to', 'protein'].edge_index = self.get_hyper_edge_index_drug_to_proteins(drug_to_proteins,
                                                                                              node_labels,
                                                                                              global_node_labels_p,
                                                                                              sparse_incidence_matrix=False)

        return data

    def save_data(self, data: HeteroData, filepath: str) -> None:
        """
        Store HeteroData object to disk
        """
        torch.save(data, filepath)

    def load_data(self, filepath: str) -> HeteroData:
        """
        Load HeteroData object from disk
        """
        return torch.load(filepath, weights_only=False)

    def data_exists(self, filepath: str) -> bool:
        """
        Check if stored data file exists
        """
        return os.path.exists(filepath)
