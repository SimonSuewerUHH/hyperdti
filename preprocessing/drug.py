import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import hypernetx as hnx
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocessing.drug_edge import get_atom_features, one_hot_encoding
from preprocessing.drug_helper import get_trade_name_from_smiles
from preprocessing.helper import prune_isolated_nodes


def draw_hypergraph(hyperedges, node_labels, name,
                    special_node=None, special_shape='*'):
    """
    Zeichnet den Hypergraphen und hebt optional einen Spezialknoten hervor.
    special_node: Index des Knotens, dem special_shape zugewiesen wird.
    special_shape: Matplotlib-Shape-String (https://matplotlib.org/stable/api/markers_api.html).
    """
    H = hnx.Hypergraph(hyperedges)
    plt.figure(figsize=(40, 24))

    # Zeichne Knoten und Kanten, erhalte Positionen
    pos = hnx.draw(
        H,
        with_node_labels=True,
        node_labels=node_labels,
        with_edge_labels=True,
        fill_edges=False,
        fill_edge_alpha=-0.9,
        layout_kwargs={
            'k': 1.0,  # Abstand zwischen Knoten (Standard ~1/√n)
            'scale': 10.0,  # Vergrößert das gesamte Layout
            'iterations': 200,
            'seed': 42
        },
        return_pos=True
    )

    # Spezialknoten nachzeichnen
    if special_node is not None and special_node in pos:
        x, y = pos[special_node]
        plt.scatter(
            [x], [y],
            s=300,
            marker=special_shape,
            edgecolors='black',
            linewidths=1.5,
            facecolors='white',
        )

    plt.title(f'Hypergraph of {name}')
    plt.show()


def create_minimal_hypergraph_with_labels(smiles, plot=True, use_trade_name=False):
    """
    Minimaler Hypergraph mit Beschriftungen:
      - Gesamt-Drug-Hyperedge (Trade Name)
      - Ringe (ring_0, ring_1, …)
      - Alle Heteroatome (hetero)
      - Positive/negative Regionen (pos_charge, neg_charge)
      - Rest-Gerüst (rest)
      - Extra-Knoten für das gesamte Drug-Molekül

    Node-Labels: AtomSymbol+Index, Edge-Labels: Hyperedge-Namen
    """
    # Molekül & Basisinfos
    mol = Chem.MolFromSmiles(smiles)
    nums_atom = mol.GetNumAtoms()

    # Trade-Name bestimmen
    canonical_smiles = Chem.MolToSmiles(mol)
    trade_name = smiles
    if use_trade_name:
        trade_name = get_trade_name_from_smiles(canonical_smiles)

    # Neuen Knoten-Index für das ganze Drug-Molekül
    drug_node = nums_atom

    hyperedges = {}
    # 0) Gesamtes Molekül als Hyperedge mit Trade Name (inkl. Drug-Knoten)
    hyperedges[trade_name] = set(range(nums_atom)) | {drug_node}

    # 1) Ringe
    for i, ring in enumerate(mol.GetRingInfo().AtomRings()):
        hyperedges[f"ring_{i}"] = set(ring)

    # 2) Alle Heteroatome
    hetero = {atom.GetIdx() for atom in mol.GetAtoms()
              if atom.GetSymbol() in ('S', 'N', 'O')}
    if len(hetero) > 1:
        hyperedges["hetero"] = hetero

    # 3) Partial Charges
    AllChem.ComputeGasteigerCharges(mol)
    pos = {i for i in range(nums_atom)
           if float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) > 0}
    neg = {i for i in range(nums_atom)
           if float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) < 0}
    if len(pos) > 1:
        hyperedges["pos_charge"] = pos
    if len(neg) > 1:
        hyperedges["neg_charge"] = neg

    # 4) Rest-Gerüst
    covered = set().union(*hyperedges.values()) if hyperedges else set()
    rest = set(range(nums_atom)) - covered
    if len(rest) > 1:
        hyperedges["rest"] = rest

    # Node-Labels: AtomSymbol+Index + Drug-Knoten
    node_labels = {i: mol.GetAtomWithIdx(i).GetSymbol()
                   for i in range(nums_atom)}
    node_labels[drug_node] = trade_name  # Sonderlabel für Drug-Knoten

    # Nodes features
    dim = get_atom_features(mol.GetAtomWithIdx(0)).shape[0]
    node_feats = np.zeros((nums_atom + 1, dim), dtype=float)
    for i in range(nums_atom):
        node_feats[i] = get_atom_features(mol.GetAtomWithIdx(i))
    node_feats[drug_node] = np.zeros(dim, dtype=float)

    # edge_features

    types = ["drug",
             "ring_aromatic", "ring_aliphatic",
             "hetero", "pos_charge", "neg_charge",
             "hbond_donor", "hbond_acceptor",
             "rest", "Unknown"]
    dim_edge = len(types)
    edge_feats = np.zeros((len(hyperedges), dim_edge), dtype=float)
    for i, (key, nodes) in enumerate(hyperedges.items()):
        base = key.split("_")[0] if key.startswith("ring_") else key
        edge_feats[i] = np.array(one_hot_encoding(base, types), dtype=int)

    if plot:
        draw_hypergraph(
            hyperedges,
            node_labels,
            name=trade_name,
            special_node=drug_node
        )

    return hyperedges, node_labels, trade_name, node_feats, edge_feats


def create_shared_hypergraph_with_labels(smiles_list, plot=True, plot_individual=False
                                         , max_similar=8, min_similar=4):
    """
    Baut aus einer Liste von SMILES‐Strings einen gemeinsamen Hypergraphen:
      - Jeder Atomic‐Knoten (AtomSymbol+Index) sowie jeder Drug‐Knoten
        wird global indiziert und gelabelt.
      - Jede lokale Hyperedge (ring_i, hetero, pos_charge, neg_charge, rest,
        Drug‐Gesamt) wird übernommen und bekommt einen eindeutigen Key
        "{smiles}_{edge_name}".
      - Falls nach dem Zusammenführen zwei oder mehr Hyperedges exakt dieselbe
        Knotenset haben, behalten wir nur eine, löschen die Duplikate und
        „redirecten" alle anderen Hyperedges, die diese Knoten referenzieren,
        auf die eine behaltene Hyperedge.
    """
    # 1) Vorbereitung
    global_hyperedges = {}  # Dict[edge_label, list[global_node_idx]]
    helper_global_hyperedges = {}  # Dict[edge_label, list[str]]
    global_node_labels = {}  # Dict[global_node_idx, str]
    offset = 0  # Laufender Zähler zur Verschiebung der lokalen Indizes
    trade_names = []

    global_node_features = []
    global_edge_features = []

    # 2) Pro SMILES: lokalen Graphen holen und in globalen ablegen
    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        try:
            local_hyperedges, local_node_labels, trade_name, local_node_features, local_nodes_labels = create_minimal_hypergraph_with_labels(
                smiles, plot=plot_individual
            )
        except Exception as e:
            print(f"Error processing SMILES '{smiles}': {e}")
            continue
        trade_names.append(trade_name)

        # 2a) Lokale Knoten → globale Knoten mappen
        local_to_global = {}
        local_to_global_labels = {}
        new_global_hyperedges = {}
        new_helper_global_hyperedges = {}
        new_helper_global_hyperedges_local_idx = {}
        already_added = []
        for local_idx, lbl in local_node_labels.items():
            global_idx = offset + local_idx
            global_node_labels[global_idx] = lbl
            global_node_features.append(local_node_features[local_idx])
            local_to_global[local_idx] = global_idx
            local_to_global_labels[local_idx] = lbl

        # 2b) Lokale Hyperedges übernehmen mit neuem Label

        for i, (edge_name, nodeset) in enumerate(local_hyperedges.items()):
            merged_label = trade_name
            if not trade_name == edge_name:
                merged_label = f"{trade_name}_{edge_name}"
            new_global_hyperedges[merged_label] = [
                local_to_global[i] for i in nodeset
            ]
            new_helper_global_hyperedges[merged_label] = [
                local_to_global_labels[i] for i in nodeset
            ]
            new_helper_global_hyperedges_local_idx[merged_label] = i

        # 2c) Offset um Anzahl der lokalen Knoten erhöhen
        offset += len(local_node_labels)

        for new_edge_label, new_edge_val in new_helper_global_hyperedges.items():
            for existing_label, existing_val in helper_global_hyperedges.items():
                if len(existing_val) > max_similar or len(existing_val) < min_similar:
                    continue
                if existing_label in existing_val:
                    continue
                if existing_val == new_edge_val:
                    already_added.append(existing_label)
                    temp_translation = {}
                    for i, node_idx in enumerate(new_global_hyperedges[new_edge_label]):
                        temp_translation[node_idx] = global_hyperedges[existing_label][i]

                    for edge_label, node_ids in new_global_hyperedges.items():
                        for i, node_idx in enumerate(node_ids):
                            if node_idx in temp_translation.keys():
                                new_global_hyperedges[edge_label][i] = temp_translation[node_idx]
                    break

            helper_global_hyperedges[new_edge_label] = new_edge_val
            global_hyperedges[new_edge_label] = new_global_hyperedges[new_edge_label]
            global_edge_features.append(
                local_nodes_labels[new_helper_global_hyperedges_local_idx[new_edge_label]]
            )

        # 3) Duplikate entfernen - nur unique Hyperedges behalten
        unique_nodesets = {}  # Dict[frozenset, edge_label]
        duplicate_edges = set()

        # Find duplicates and keep first occurrence
        for edge_label, nodes in global_hyperedges.items():
            nodeset = frozenset(nodes)
            if nodeset in unique_nodesets:
                duplicate_edges.add(edge_label)
            else:
                unique_nodesets[nodeset] = edge_label

        # Remove duplicates
        for edge_label in duplicate_edges:
            edge_labels = list(global_hyperedges.keys())
            del_index = edge_labels.index(edge_label)
            del global_hyperedges[edge_label]
            del global_edge_features[del_index]
    # 4) Plotten
    if plot:
        draw_hypergraph(
            global_hyperedges,
            global_node_labels,
            name="Shared Minimal Hypergraph"
        )

    # 5) Rückgabe
    #6) Prune isolated nodes
    global_node_labels, global_hyperedges, global_node_features = prune_isolated_nodes(
        global_node_labels, global_hyperedges, global_node_features
    )
    return global_node_labels, global_hyperedges, trade_names, global_node_features, global_edge_features



