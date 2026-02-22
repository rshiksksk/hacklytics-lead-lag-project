import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from config import DATA_DIR, BATCH_SIZE, N_CLASSES
from dataset import HTGNNDataset
from model import HTGNN


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch.y.view(-1).cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=-1).numpy()
    labels = labels.numpy()
    probs = torch.softmax(logits, dim=-1).numpy()

    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # AUC: one-vs-rest for each class
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "macro_f1": f1, "macro_auc": auc}


def event_study(logits: torch.Tensor, labels: torch.Tensor, dataset: HTGNNDataset, horizon: int, n_days: int = 40):
    """
    For each predicted outperformer (class 2) and underperformer (class 0),
    track demeaned returns for the next n_days and compute average CARs.
    """
    preds = logits.argmax(dim=-1).numpy()
    dret = dataset.feat_matrix[:, :, 0]  # (n_stocks, n_dates)
    n_d = dret.shape[1]

    car_out = np.zeros(n_days)
    car_under = np.zeros(n_days)
    count_out = count_under = 0

    for i, (si, di) in enumerate(dataset.samples):
        if di + n_days >= n_d:
            continue
        future_rets = dret[si, di + 1: di + 1 + n_days]  # (n_days,)
        if np.any(np.isnan(future_rets)):
            continue
        cumulative = np.cumsum(future_rets)

        if preds[i] == 2:  # predicted outperform
            car_out += cumulative
            count_out += 1
        elif preds[i] == 0:  # predicted underperform
            car_under += cumulative
            count_under += 1

    if count_out > 0:
        car_out /= count_out
    if count_under > 0:
        car_under /= count_under

    days = np.arange(1, n_days + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(days, car_out   * 100, label=f"Predicted Outperform (n={count_out})",   color="green")
    plt.plot(days, car_under * 100, label=f"Predicted Underperform (n={count_under})", color="red")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Days after signal")
    plt.ylabel("Cumulative Abnormal Return (%)")
    plt.title(f"Event Study — Horizon h={horizon}")
    plt.legend()
    plt.tight_layout()
    out = DATA_DIR / f"event_study_h{horizon}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Event study plot saved to {out}")

    return car_out, car_under, count_out, count_under


def run(horizon: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Evaluating h={horizon} ===")

    test_ds = HTGNNDataset(horizon, "test")
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HTGNN().to(device)
    model.load_state_dict(torch.load(DATA_DIR / f"model_h{horizon}.pt", map_location=device))

    logits, labels = get_predictions(model, loader, device)
    metrics = compute_metrics(logits, labels)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"Macro AUC: {metrics['macro_auc']:.4f}")

    car_out, car_under, n_out, n_under = event_study(logits, labels, test_ds, horizon)
    print(f"Event study: {n_out} outperform signals, {n_under} underperform signals")
    print(f"  CAR[+5d] outperform  : {car_out[4]*100:+.3f}%")
    print(f"  CAR[+5d] underperform: {car_under[4]*100:+.3f}%")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[1, 3, 5, 20], required=True)
    args = parser.parse_args()
    run(args.horizon)
