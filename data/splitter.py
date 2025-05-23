import torch
from torch_geometric.data import HeteroData


def split_masks(
    data: HeteroData,
    target_node_type: str = 'paper',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Attach .train/.val/.test_mask to data[target_node_type]."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    num = data[target_node_type].num_nodes
    perm = torch.randperm(num, generator=torch.Generator().manual_seed(seed))
    n_train = int(train_ratio * num)
    n_val   = int(val_ratio   * num)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]
    mask = lambda idx: torch.zeros(num, dtype=torch.bool, device=data[target_node_type].x.device).index_fill_(0, idx, True)
    data[target_node_type].train_mask = mask(train_idx)
    data[target_node_type].val_mask   = mask(val_idx)
    data[target_node_type].test_mask  = mask(test_idx)