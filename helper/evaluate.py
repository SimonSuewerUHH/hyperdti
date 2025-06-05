import torch
from torch_geometric.data import HeteroData


def split_edge_train_val_test(
        data: HeteroData,
        edge_type: tuple = ('drug', 'to', 'protein'),
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
) -> HeteroData:
    """
    Splits positive edges of the given edge_type into train/val/test splits.
    Modifies data in-place by setting:
      data[edge_type].edge_index       -> training edges
      data[edge_type].val_pos_edge_index
      data[edge_type].test_pos_edge_index
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1"
    edges = data[edge_type].edge_index
    num_edges = edges.size(1)
    n_train = int(train_ratio * num_edges)
    n_val = int(val_ratio * num_edges)

    train = edges[:n_train]
    val = edges[n_train:n_train + n_val]
    test = edges[n_train + n_val:]

    # store splits
    data[edge_type].val_pos_edge_index = val
    data[edge_type].test_pos_edge_index = test
    # restrict main edge_index to train only
    data[edge_type].edge_index = train

    return data


def evaluate(
        model: torch.nn.Module,
        data: HeteroData,
        split: str = 'val',
        device: torch.device = None
) -> float:
    """
    Evaluates the model on the given split ('train','val','test') using only positive edges.
    Returns the fraction of positives predicted above threshold (0).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    edge_type = ('drug', 'to', 'protein')
    # select positive edges
    if split == 'train':
        pos_edge_index = data[edge_type].edge_index
    elif split == 'val':
        pos_edge_index = data[edge_type].val_pos_edge_index
    elif split == 'test':
        pos_edge_index = data[edge_type].test_pos_edge_index
    else:
        raise ValueError(f"Unknown split: {split}")

    # compute logits for positives only
    with torch.no_grad():
        logits = model(data, pos_edge_index=pos_edge_index)
    return get_recall(logits)


def get_recall(logits):
    if isinstance(logits, float):
        return logits

    logits = logits.view(-1).cpu()

    # compute proportion above zero threshold (recall)
    preds = (logits >= 0).float()
    recall = preds.mean().item()
    return recall


def test_model(
        model: torch.nn.Module,
        data: HeteroData,
        device: torch.device = None
) -> None:
    """
    Tests the model on the test split and prints recall on positive edges.
    """
    test_recall = evaluate(model, data, split='test', device=device)
    print(f"Test Recall (Positives): {test_recall:.4%}")
