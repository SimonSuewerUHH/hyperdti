import itertools


def prune_isolated_nodes(global_node_labels, global_hyperedges, global_node_features):
    """
    Entfernt alle Knoten, die in keinem Hyperedge referenziert werden,
    und kompakt die verbleibenden Knoten-Indizes von 0 bis N-1:
      - global_node_labels: Dict[old_idx -> label]
      - global_hyperedges: Dict[edge_label -> list[old_idx]]
      - global_node_features: List[np.array] oder np.ndarray, indexiert nach old_idx

    Rückgabe:
      new_node_labels: Dict[new_idx -> label]
      new_hyperedges: Dict[edge_label -> list[new_idx]]
      new_node_features: List der Features in neuer Reihenfolge
    """
    # 1) Bestimme alle tatsächlich genutzten alten Knoten-Indizes
    used_nodes = set(itertools.chain.from_iterable(global_hyperedges.values()))

    # 2) Erstelle Mapping old_idx -> new_idx (kompakt, sortiert)
    sorted_used = sorted(used_nodes)
    old_to_new = {old: new for new, old in enumerate(sorted_used)}

    # 3) Baue neue node_labels und node_features in Reihenfolge new_idx auf
    new_node_labels = {
        old_to_new[old]: global_node_labels[old]
        for old in sorted_used
    }
    new_node_features = [
        global_node_features[old] for old in sorted_used
    ]

    # 4) Übersetze Hyperedges auf neue Indizes
    new_hyperedges = {}
    for edge_name, nodes in global_hyperedges.items():
        # Nur die in used_nodes liegenden; sollte immer der Fall sein
        new_nodes = [old_to_new[old] for old in nodes if old in old_to_new]
        new_hyperedges[edge_name] = new_nodes

    return new_node_labels, new_hyperedges, new_node_features