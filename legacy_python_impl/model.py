import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from config import ROLLING_WINDOW, NODE_FEAT_DIM, HIDDEN_DIM, N_CLASSES, EDGE_DIM, DROPOUT


class EdgeWeightedSAGE(MessagePassing):
    """GraphSAGE layer that incorporates edge features during aggregation."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = EDGE_DIM):
        super().__init__(aggr="mean")
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_self(x) + out
        return F.relu(self.bn(out))

    def message(self, x_j, edge_attr):
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))


class HTGNN(nn.Module):
    def __init__(
        self,
        feat_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_classes: int = N_CLASSES,
        T: int = ROLLING_WINDOW,
        edge_dim: int = EDGE_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.T = T
        self.feat_dim = feat_dim
        self.sage = EdgeWeightedSAGE(feat_dim, hidden_dim, edge_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, data):
        # data.x: (total_nodes, T * feat_dim)
        # data.ptr: cumulative node counts per graph in batch
        x = data.x.view(data.x.shape[0], self.T, self.feat_dim)  # (total_nodes, T, feat_dim)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # First node of each subgraph = focal node
        focal_idx = data.ptr[:-1]  # (batch_size,)

        # Run SAGE at each timestep, collect focal node output
        h_seq = []
        for t in range(self.T):
            h_t = self.sage(x[:, t, :], edge_index, edge_attr)  # (total_nodes, hidden)
            h_seq.append(h_t[focal_idx])                         # (batch_size, hidden)

        h_seq = torch.stack(h_seq, dim=1)        # (batch_size, T, hidden)
        h_seq = self.dropout(h_seq)
        _, (h_n, _) = self.lstm(h_seq)           # h_n: (1, batch_size, hidden)
        out = self.classifier(self.dropout(h_n.squeeze(0)))  # (batch_size, n_classes)
        return out
