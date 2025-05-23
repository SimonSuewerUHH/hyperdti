import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import HeteroData
from tqdm import tqdm

from data.splitter import split_masks
from helper.evaluate import evaluate, test
from helper.parser import Config
from model.model import HeteroHyperModel


def train_epoch(model, data: HeteroData, optimizer, criterion) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    edge_index = torch.ones(logits.shape[0])
    loss = criterion(logits,edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(data: HeteroData, cfg: Config):
    # split_masks(
    #    data,
    #    target_node_type='paper',
    #    train_ratio=cfg.train_ratio,
    #    val_ratio=cfg.val_ratio,
    #    test_ratio=cfg.test_ratio,
    #    seed=cfg.seed,
    # )
    data = data.to(cfg.device)

    # --- Model, optimizer, scheduler, loss --------------------------------
    model = HeteroHyperModel(
        drug_in=data['drug_atom'].x.size(1),
        drug_edge=data['drugs_hyperedge'].x.size(1),
        protein_in=data['proteins'].x.size(1),
        hidden=cfg.hidden_dim,
        heads=cfg.heads,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in tqdm(range(1, cfg.epochs + 1), desc='Epochs'):
        loss = train_epoch(model, data, optimizer, criterion)
        # train_acc = evaluate(model, data, split='train')
        # val_acc = evaluate(model, data, split='val')
        tqdm.write(f"Epoch {epoch}/{cfg.epochs} | Loss: {loss:.4f} | ")
        #           f"Train Acc: {train_acc:.4%} | Val Acc: {val_acc:.4%}")
        scheduler.step()

    # Final test
    test_acc = evaluate(model, data, split='test')
    print(f"Test Acc: {test_acc:.4%}")
