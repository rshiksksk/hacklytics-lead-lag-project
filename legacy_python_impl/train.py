import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from config import DATA_DIR, HIDDEN_DIM, N_CLASSES, BATCH_SIZE, LR, EPOCHS, PATIENCE
from dataset import HTGNNDataset
from model import HTGNN


def get_class_weights(dataset: HTGNNDataset) -> torch.Tensor:
    labels = [dataset.get(i).y.item() for i in range(len(dataset))]
    weights = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=labels)
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        correct += (logits.argmax(dim=-1) == batch.y.view(-1)).sum().item()
        n += batch.num_graphs
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        total_loss += loss.item() * batch.num_graphs
        correct += (logits.argmax(dim=-1) == batch.y.view(-1)).sum().item()
        n += batch.num_graphs
    return total_loss / n, correct / n


def run(horizon: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Horizon h={horizon} | device={device} ===")

    train_ds = HTGNNDataset(horizon, "train")
    val_ds   = HTGNNDataset(horizon, "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HTGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    class_weights = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    patience_counter = 0
    best_path = DATA_DIR / f"model_h{horizon}.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best model saved to {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[1, 3, 5, 20], required=True)
    args = parser.parse_args()
    run(args.horizon)
