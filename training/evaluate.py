import torch

def evaluate(model, data, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        data = data.to(device)
        out_data = model(data)
        for batch_ids, batch_labels in loader:
            batch_ids = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            embeddings = out_data['hyperedge'].x[batch_ids]
            logits = model.mlp(embeddings).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            total_loss += loss.item() * batch_ids.size(0)

            preds = torch.sigmoid(logits).cpu()
            all_preds.append(preds)
            all_labels.append(batch_labels.cpu())
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(labels, preds)
    return avg_loss, auc