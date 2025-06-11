import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import HeteroData
from tqdm import tqdm

from helper.evaluate import evaluate, split_edge_train_val_test, test_model, get_recall
from helper.negative_sampling import hetero_negative_sampling
from helper.parser import Config
from model.model import HeteroHyperModel


def train_epoch(
    model,
    data: HeteroData,
    optimizer,
    criterion,
    batch_size: int = 800,
    neg_ratio: float = 0.1
) -> float:
    """
    Performs one epoch of training in mini-batches for link prediction.
    """
    model.train()
    total_loss = 0.0

    # 1) Get all positive edges of type ('drug','to','protein')
    pos_edge_index = data['drug', 'to', 'protein'].edge_index  # shape: [2, num_pos]
    num_pos = pos_edge_index.size(1)

    # 2) Create a random permutation of indices [0 .. num_pos-1]
    perm = torch.randperm(num_pos, device=pos_edge_index.device)

    # 3) Loop over mini‐batches
    num_batches = 0
    for start in range(0, num_pos, batch_size):
        end = min(start + batch_size, num_pos)
        batch_idx = perm[start:end]
        pos_batch = pos_edge_index[:, batch_idx]  # shape [2, batch_size_current]

        num_pos_batch = pos_batch.size(1)
        if num_pos_batch == 0:
            continue

        # 4) Negative Sampling: sample num_pos_batch * neg_ratio negative edges
        #    (you can cast to int; if you want exactly same number, ensure int conversion)
        num_neg_batch = int(num_pos_batch * neg_ratio)
        neg_batch = hetero_negative_sampling(
            pos_edge_index=pos_batch,
            num_neg=num_neg_batch
        )  # shape [2, num_neg_batch]

        # 5) Build labels: ones for positive batch, zeros for negative batch
        y_pos = torch.ones(num_pos_batch, device=pos_edge_index.device)
        y_neg = torch.zeros(neg_batch.size(1), device=pos_edge_index.device)
        labels = torch.cat([y_pos, y_neg], dim=0)  # shape [num_pos_batch + num_neg_batch]

        # 6) Forward + backward on this mini‐batch
        optimizer.zero_grad()
        logits = model(
            data,
            pos_edge_index=pos_batch,
            neg_edge_index=neg_batch
        )  # shape: [num_pos_batch + num_neg_batch]

        loss = criterion(logits.view(-1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # 7) Return average loss over all batches
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


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
        train_recall = get_recall(loss)
        #val_recall = evaluate(model, data, split='val')
        print( f" | Train Recall: {train_recall:.4%}")
        #print(f" | Val Recall: {val_recall:.4%}")
        tqdm.write(
            f"Epoch {epoch}/{cfg.epochs} | Loss: {loss:.4f}"
            f" | Train Recall: {train_recall:.4%}"
            #f" | Val Recall: {val_recall:.4%}"
        )
        scheduler.step()

    # Final test
    test_model(model, data, device=cfg.device)
