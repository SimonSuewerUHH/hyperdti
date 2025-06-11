import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import HeteroData
from tqdm import tqdm

from helper.evaluate import split_edge_train_val_test, test_model, get_recall
from helper.negative_sampling import hetero_negative_sampling
from helper.parser import Config
from model.model import HeteroHyperModel


def train_epoch(model,
                data: HeteroData,
                optimizer,
                criterion,
                neg_ratio: float = 0.1) -> float:
    model.train()
    optimizer.zero_grad()

    # 1) Positive Kanten
    pos_edge_index = data['drug', 'to', 'protein'].edge_index
    num_pos = pos_edge_index.size(1)

    # 2) Negative Sampling: gleiche Anzahl wie Positiv
    neg_edge_index = hetero_negative_sampling(
        pos_edge_index=pos_edge_index,
        num_neg=int(num_pos * neg_ratio))

    # 3) Labels erstellen
    y_pos = torch.ones(num_pos, device=pos_edge_index.device)
    y_neg = torch.zeros(neg_edge_index.size(1), device=pos_edge_index.device)
    labels = torch.cat([y_pos, y_neg], dim=0)

    # 4) Modell-Vorwärtslauf mit beiden Kanten
    logits = model(data, pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index)

    loss = criterion(logits.view(-1), labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(data: HeteroData, cfg: Config):
    data = split_edge_train_val_test(
        data,
        edge_type=('drug', 'to', 'protein'),
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed
    )
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
        #train_recall = get_recall(loss)
        # val_recall = evaluate(model, data, split='val')
        print(f"Train Loss: {loss:.4%}")
        # print(f" | Val Recall: {val_recall:.4%}")
        #tqdm.write(
        #    f"Epoch {epoch}/{cfg.epochs} | Loss: {loss:.4f}"
        #    f" | Train Recall: {train_recall:.4%}"
            # f" | Val Recall: {val_recall:.4%}"
        #)
        scheduler.step()

    # Final test
    test_model(model, data, device=cfg.device)
