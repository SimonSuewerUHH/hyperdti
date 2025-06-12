import os

import numpy as np
import torch
import esm
import hypernetx as hnx
import matplotlib.pyplot as plt
import re

from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer, pipeline

from preprocessing.helper import prune_isolated_nodes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd',
                                          do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd",
                                        ignore_mismatched_sizes=True).to(device)

def get_model():
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    return model, alphabet


def check_cache(prot_id: str):
    cache_path = f"cache/{prot_id}_contacts.npy"
    if os.path.exists(cache_path):
        print(f"Loading contact map from cache: {prot_id}")
        return np.load(cache_path)
    return None

def save_cache(prot_id: str, contact_map):
    cache_path = f"cache/{prot_id}_contacts.npy"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, contact_map)
    print(f"Saved contact map to cache: {prot_id}")

def get_protbert_embedding(seq):
    sequence = " ".join(list(re.sub(r"[UZOB]", "X", seq)))

    ids = tokenizer(sequence, return_tensors='pt')
    input_ids = ids['input_ids'].clone().detach().to(device)
    attention_mask = ids['attention_mask'].clone().detach().to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    output1 = output[0][0][1: -1]

    assert len(seq) == len(output1)

    return output1.cpu()


def prune_hypergraph(hyperedges, node_labels, node_values):
    """
    Prunes and reindexes a hypergraph so that node indices run from 1 to n without gaps.

    Args:
        hyperedges: dict mapping edge_id to tuple of original node indices
        node_labels: dict mapping original node index to label
        node_values: np.ndarray of shape (num_nodes, embedding_dim)

    Returns:
        pruned_edges: dict mapping edge_id to tuple of new node indices (1-based)
        pruned_labels: dict mapping new node index to label
        pruned_values: np.ndarray of shape (n, embedding_dim)
    """
    # Collect all nodes referenced in hyperedges
    all_nodes = sorted({node for nodes in hyperedges.values() for node in nodes})
    # Create mapping from old index to new consecutive index (1-based)
    mapping = {old: new for new, old in enumerate(all_nodes, start=0)}

    # Reindex hyperedges
    pruned_edges = {
        edge_id: tuple(mapping[node] for node in nodes)
        for edge_id, nodes in hyperedges.items()
    }
    # Reindex labels
    pruned_labels = {
        mapping[node]: label
        for node, label in node_labels.items()
        if node in mapping
    }
    # Prune node_values rows
    pruned_values = node_values[all_nodes, :]

    return pruned_edges, pruned_labels, pruned_values


def build_local_protein_hypergraph(contact_map: np.ndarray,
                                   sequence: str,
                                   threshold: float = 0.5,
                                   trade_name: str = "",
                                   combine_nodes: bool = True,
                                   plot: bool = False):  # no plotting
    """
    Builds a local protein hypergraph from contact probabilities.

    Args:
        contact_map: L×L matrix of contact probabilities.
        sequence: Protein sequence of length L.
        threshold: Probability cutoff.
        trade_name: Protein identifier for edge labels.

    Returns:
        local_hyperedges: Mapping edge_id → tuple of residue indices.
        local_node_labels: Mapping residue index → amino acid letter.
    """
    L = contact_map.shape[0]
    seen_sets = set()
    hyperedges = {}
    edge_counter = 0
    # Build hyperedges
    for i in range(L):
        partners = np.where(contact_map[i] >= threshold)[0].tolist()
        if partners:
            nodes = tuple(sorted({i, *partners}))
            fs = frozenset(nodes)
            if fs not in seen_sets:
                seen_sets.add(fs)
                hyperedges[f"{trade_name}_he_{edge_counter}"] = nodes
                edge_counter += 1

    # Get node embeddings and add an extra row for the global protein node
    node_values = get_protbert_embedding(sequence).numpy()
    # Append empty embedding for the global node
    empty_row = np.zeros((1, node_values.shape[1]))
    node_values = np.vstack([node_values, empty_row])

    # 2) Add a global hyperedge spanning only residues that appear in any contact (when combine_nodes)
    nodes_with_edges = set()
    if combine_nodes:
        for nodes in hyperedges.values():
            nodes_with_edges.update(nodes)
    else:
        nodes_with_edges = range(L)
    all_nodes = tuple(sorted(nodes_with_edges))
    if all_nodes and frozenset(all_nodes) not in seen_sets:
        hyperedges[trade_name] = all_nodes
        edge_counter += 1
    # global protein node
    node_labels = {i: sequence[i] for i in range(L)}
    node_labels[L] = trade_name
    hyperedges[trade_name] = hyperedges[trade_name] + (L,)
    # 4) Optional visualization with customizable draw parameters
    pruned_edges, pruned_labels, pruned_values = prune_hypergraph(
        hyperedges, node_labels, node_values
    )
    if plot:
        H = hnx.Hypergraph(pruned_edges)
        plt.figure(figsize=(40, 24))  # Increased figure size for better readability
        hnx.draw(
            H,
            with_node_labels=True,
            node_labels=pruned_labels,
            with_edge_labels=True,
        )
        plt.title(f"Hypergraph of {trade_name}")
        plt.show()
    return pruned_edges, pruned_labels, pruned_values


def create_shared_protein_hypergraph(proteins,
                                     model,
                                     alphabet,
                                     threshold: float = 0.5,
                                     plot: bool = True,
                                     use_cache: bool = True):
    """
    Merges multiple protein hypergraphs into a shared global hypergraph.

    Args:
        proteins: List of (prot_id, sequence) tuples.
        model: ESM model for contact prediction.
        alphabet: Corresponding alphabet for batch converter.
        threshold: Contact probability cutoff.
        plot: Whether to visualize the final shared hypergraph.

    Returns:
        global_node_labels: Mapping global node idx → label.
        global_hyperedges: Mapping edge label → list of global node indices.
    """
    batch_converter = alphabet.get_batch_converter()
    global_hyperedges = {}
    helper_labels = {}  # for duplication detection (node symbols)
    global_node_labels = {}
    offset = 0
    sawn_proteins = set()
    global_node_features = []

    # 1) Iterate proteins
    for prot_id, sequence in tqdm(proteins, desc="Processing proteins"):
        if prot_id in sawn_proteins:
            continue
        sawn_proteins.add(prot_id)
        # Contact prediction

        contact_map = None
        if use_cache:
            contact_map = check_cache(prot_id) #with no cache always None
        if contact_map is None:
            _, _, batch_tokens = batch_converter([(prot_id, sequence)])
            with torch.no_grad():
                try:
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                except ValueError as e:
                    print(f"Skipping {prot_id} due to error: {e}")
                    continue
            contact_map = results["contacts"][0].cpu().numpy()
            if use_cache:
                save_cache(prot_id, contact_map)
        # Local hypergraph
        local_edges, local_nodes, local_node_features = build_local_protein_hypergraph(
            contact_map, sequence, threshold, trade_name=prot_id
        )
        # Map to global indices
        local_to_global = {}
        for local_idx, aa in local_nodes.items():
            gidx = offset + local_idx
            global_node_labels[gidx] = f"{aa}{local_idx}_{prot_id}"
            global_node_features.append(local_node_features[local_idx])
            local_to_global[local_idx] = gidx
        offset += len(local_nodes)
        # Add edges
        for edge_label, nodes in local_edges.items():
            global_nodes = [local_to_global[n] for n in nodes]
            # use node features for duplication test 
            node_features = [local_node_features[n] for n in nodes]
            node_features_tuple = tuple(map(tuple, node_features))
            # Deduplication
            duplicate = False
            for exist_lbl, exist_nodes in global_hyperedges.items():
                exist_features = [global_node_features[n] for n in exist_nodes]
                exist_features_tuple = tuple(map(tuple, exist_features))
                if set(node_features_tuple) == set(exist_features_tuple):
                    duplicate = True
                    break
            if not duplicate:
                global_hyperedges[f"{edge_label}"] = global_nodes
                helper_labels[f"{edge_label}"] = node_features
    # 2) Optional plot
    if plot:
        H = hnx.Hypergraph(global_hyperedges)
        plt.figure(figsize=(20, 12))
        hnx.draw(
            H,
            with_node_labels=True,
            node_labels=global_node_labels,
            with_edge_labels=True,
        )
        plt.title("Shared Protein Hypergraph")
        plt.show()

    global_node_labels, global_hyperedges, global_node_features = prune_isolated_nodes(
        global_node_labels, global_hyperedges, global_node_features
    )

    return global_node_labels, global_hyperedges, global_node_features
