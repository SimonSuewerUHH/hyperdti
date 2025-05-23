import torch.nn.functional as F

def train_one_epoch(model, data, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_ids, batch_labels in loader:
        batch_ids = batch_ids.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        # Forward auf kompletten Graphen
        data = data.to(device)
        out_data = model(data)
        # Nur die relevanten Hyperedge-Embeddings
        embeddings = out_data['hyperedge'].x[batch_ids]
        logits = model.mlp(embeddings).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_ids.size(0)
    return total_loss / len(loader.dataset)