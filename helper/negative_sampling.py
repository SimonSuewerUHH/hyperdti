import torch
from torch import Tensor


def hetero_negative_sampling(pos_edge_index: Tensor, num_neg: int):
    # pos_edge_index: [2, num_pos], num_nodes=(#drug_nodes, #prot_nodes)
    # 1) Finde alle tatsächlich verwendeten Drugs und Proteine
    drugs = pos_edge_index[0].unique()
    prots = pos_edge_index[1].unique()

    # 2) Erzeuge den vollständigen kartesischen Produkt-Raum Drug×Protein
    d = drugs.view(-1, 1).repeat(1, prots.size(0)).view(-1)
    p = prots.repeat(drugs.size(0))
    all_pairs = torch.stack([d, p], dim=0)  # [2, |drugs|*|prots|]

    # 3) Baue ein Set mit den positiven Kanten zur schnellen Prüfung
    pos_set = set([(int(i), int(j)) for i, j in pos_edge_index.t().tolist()])

    # 4) Filtere das kartesische Produkt heraus, was schon positiv ist
    mask = torch.tensor(
        [(int(all_pairs[0, k]), int(all_pairs[1, k])) not in pos_set
         for k in range(all_pairs.size(1))],
        dtype=torch.bool, device=all_pairs.device
    )
    neg_candidates = all_pairs[:, mask]

    # 5) Ziehe zufällig num_neg Samples
    if neg_candidates.size(1) < num_neg:
        raise ValueError(f"Zu wenige negative Kandidaten: {neg_candidates.size(1)} < {num_neg}")
    perm = torch.randperm(neg_candidates.size(1), device=all_pairs.device)[:num_neg]
    return neg_candidates[:, perm]
