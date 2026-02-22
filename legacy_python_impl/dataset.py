import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from config import (
    DATA_DIR, ROLLING_WINDOW, FINANCIAL_DIM, EMBEDDING_DIM,
    NODE_FEAT_DIM, LABEL_K, TRAIN_END, VAL_END,
)
from supply_chain import get_snapshot_year


class HTGNNDataset(Dataset):
    def __init__(self, horizon: int, split: str):
        super().__init__()
        assert split in ("train", "val", "test")
        self.horizon = horizon
        self.split = split
        self.T = ROLLING_WINDOW

        features = pd.read_parquet(DATA_DIR / "daily_features.parquet")
        features["Trddt"] = pd.to_datetime(features["Trddt"])

        with open(DATA_DIR / "supply_chain_graphs.pkl", "rb") as f:
            self.graphs = pickle.load(f)
        with open(DATA_DIR / "node2vec_embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)

        self.stocks = sorted(features["Stkcd"].unique())
        self.stock_to_idx = {s: i for i, s in enumerate(self.stocks)}

        self.dates = sorted(features["Trddt"].unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}

        n_s, n_d = len(self.stocks), len(self.dates)
        self.feat_matrix = np.full((n_s, n_d, FINANCIAL_DIM), np.nan, dtype=np.float32)
        si_arr = features["Stkcd"].map(self.stock_to_idx).values
        di_arr = features["Trddt"].map(self.date_to_idx).values
        self.feat_matrix[si_arr, di_arr] = features[
            ["demeaned_ret", "log_mktcap", "abn_turnover", "LimitStatus"]
        ].values

        self.fwd_returns = self._compute_forward_returns()
        self.thresholds = self._compute_thresholds()

        # Precompute adjacency lists for fast neighbor lookup
        self.adj = self._build_adj()

        self.samples = self._build_samples()
        print(f"[{split}] horizon={horizon}: {len(self.samples)} samples")

    # ------------------------------------------------------------------
    def _compute_forward_returns(self) -> np.ndarray:
        h = self.horizon
        dret = self.feat_matrix[:, :, 0]  # (n_stocks, n_dates)
        n_s, n_d = dret.shape
        fwd = np.full((n_s, n_d), np.nan, dtype=np.float32)
        for t in range(n_d - h):
            fwd[:, t] = np.nansum(dret[:, t + 1: t + 1 + h], axis=1)
        return fwd

    def _compute_thresholds(self) -> np.ndarray:
        """Expanding std of realized h-day forward returns up to each date."""
        h = self.horizon
        n_d = self.fwd_returns.shape[1]
        thresholds = np.full(n_d, np.nan, dtype=np.float32)
        running_sum = 0.0
        running_sq = 0.0
        running_n = 0
        for t in range(n_d):
            if t >= h:
                realized = self.fwd_returns[:, t - h]
                valid = realized[~np.isnan(realized)]
                running_sum += valid.sum()
                running_sq += (valid ** 2).sum()
                running_n += len(valid)
            if running_n > 1:
                mean = running_sum / running_n
                var = max(running_sq / running_n - mean ** 2, 0.0)
                thresholds[t] = LABEL_K * np.sqrt(var)
        return thresholds

    def _build_adj(self) -> dict:
        """Precompute adjacency: {year: {node_key: [(neighbor_key, rel, rank)]}}"""
        adj = {}
        for year, G in self.graphs.items():
            year_adj = {}
            for node in G.nodes():
                neighbors = []
                for succ in G.successors(node):
                    d = G[node][succ]
                    neighbors.append((succ, d["relation"], d["rank"]))
                year_adj[node] = neighbors
            adj[year] = year_adj
        return adj

    def _build_samples(self) -> list:
        T, h = self.T, self.horizon
        n_s, n_d = len(self.stocks), len(self.dates)
        dates_arr = np.array(self.dates)

        train_end = pd.Timestamp(TRAIN_END)
        val_end = pd.Timestamp(VAL_END)
        mask = {
            "train": dates_arr <= train_end,
            "val":   (dates_arr > train_end) & (dates_arr <= val_end),
            "test":  dates_arr > val_end,
        }[self.split]

        valid_di = np.where(
            mask
            & ~np.isnan(self.thresholds)
            & (np.arange(n_d) >= T - 1)
            & (np.arange(n_d) < n_d - h)
        )[0]

        # Apply stride=h (non-overlapping forward windows)
        valid_di = valid_di[::h]

        samples = []
        for di in valid_di:
            window_ok = ~np.any(np.isnan(self.feat_matrix[:, di - T + 1: di + 1, :]), axis=(1, 2))
            fwd_ok = ~np.isnan(self.fwd_returns[:, di])
            for si in np.where(window_ok & fwd_ok)[0]:
                samples.append((int(si), int(di)))
        return samples

    # ------------------------------------------------------------------
    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        si, di = self.samples[idx]
        stock = self.stocks[si]
        date = self.dates[di]
        T = self.T

        snap_year = get_snapshot_year(date)
        year_adj = self.adj.get(snap_year, {})
        year_emb = self.embeddings.get(snap_year, {})

        focal_key = f"listed_{stock}"
        neighbors = year_adj.get(focal_key, [])  # [(neighbor_key, rel, rank)]

        all_nodes = [focal_key] + [n for n, _, _ in neighbors]
        node_to_sub = {n: i for i, n in enumerate(all_nodes)}
        n_nodes = len(all_nodes)

        # Node features: (n_nodes, T * NODE_FEAT_DIM)
        x = np.zeros((n_nodes, T * NODE_FEAT_DIM), dtype=np.float32)
        for ni, nkey in enumerate(all_nodes):
            emb = year_emb.get(nkey, np.zeros(EMBEDDING_DIM, dtype=np.float32))
            if nkey.startswith("listed_"):
                nstock = int(nkey.split("_")[1])
                nsi = self.stock_to_idx.get(nstock)
                fin = self.feat_matrix[nsi, di - T + 1: di + 1] if nsi is not None else np.zeros((T, FINANCIAL_DIM), dtype=np.float32)
            else:
                fin = np.zeros((T, FINANCIAL_DIM), dtype=np.float32)

            for t in range(T):
                x[ni, t * NODE_FEAT_DIM: t * NODE_FEAT_DIM + FINANCIAL_DIM] = fin[t]
                x[ni, t * NODE_FEAT_DIM + FINANCIAL_DIM: (t + 1) * NODE_FEAT_DIM] = emb

        # Edges
        edge_src, edge_dst, edge_attrs = [], [], []
        for n_src, (n_dst_key, rel, rank) in zip(
            [focal_key] * len(neighbors), neighbors
        ):
            if n_dst_key in node_to_sub:
                edge_src.append(node_to_sub[n_src])
                edge_dst.append(node_to_sub[n_dst_key])
                edge_attrs.append([float(rel), float(rank)])
                # reverse edge
                edge_src.append(node_to_sub[n_dst_key])
                edge_dst.append(node_to_sub[n_src])
                edge_attrs.append([float(3 - rel), float(rank)])

        if edge_src:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)

        # Label
        fwd_ret = float(self.fwd_returns[si, di])
        thresh = float(self.thresholds[di])
        if fwd_ret > thresh:
            label = 2
        elif fwd_ret < -thresh:
            label = 0
        else:
            label = 1

        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=n_nodes,
        )
