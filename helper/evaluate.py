import torch
from torch_geometric.data import HeteroData


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: HeteroData,
    split: str = 'val',
    target_node_type: str = 'paper',
) -> float:
    model.eval()
    out  = model(data.x_dict, data.edge_index_dict)[target_node_type]
    y    = data[target_node_type].y
    mask = getattr(data[target_node_type], f'{split}_mask')
    preds = out[mask].argmax(dim=1)
    correct = (preds == y[mask]).sum().item()
    total   = int(mask.sum().item())
    return correct / total if total > 0 else 0.0

def test(
    model: torch.nn.Module,
    data: HeteroData,
    target_node_type: str = 'paper',
) -> float:
    return evaluate(model, data, split='test', target_node_type=target_node_type)