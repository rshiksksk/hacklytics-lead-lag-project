from pathlib import Path

DATA_DIR = Path("/Users/zaynguo/Documents/MSA/Hacklytics")

# Feature engineering
TURNOVER_WINDOW = 20
START_DATE = "2016-05-01"

# Node2Vec
EMBEDDING_DIM = 64
WALK_LENGTH = 30
NUM_WALKS = 200
N2V_WINDOW = 10
N2V_WORKERS = 4

# Model
ROLLING_WINDOW = 20       # T: number of past days fed into LSTM
FINANCIAL_DIM = 4         # demeaned_ret, log_mktcap, abn_turnover, LimitStatus
NODE_FEAT_DIM = FINANCIAL_DIM + EMBEDDING_DIM  # 68
HIDDEN_DIM = 128
N_CLASSES = 3
EDGE_DIM = 2              # relation type + rank
DROPOUT = 0.3

# Label thresholding
LABEL_K = 0.5             # outperform if fwd_ret > K * expanding_std

# Training
HORIZONS = [1, 3, 5, 20]
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
PATIENCE = 10
