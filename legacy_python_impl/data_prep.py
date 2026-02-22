import pandas as pd
import numpy as np
from config import DATA_DIR, TURNOVER_WINDOW, START_DATE


def load_and_merge() -> pd.DataFrame:
    liq = pd.read_csv(DATA_DIR / "LIQ_TOVER_D_combined.csv", parse_dates=["Trddt"])
    trd = pd.read_csv(DATA_DIR / "TRD_Dalyr_combined.csv", parse_dates=["Trddt"])
    df = liq.merge(trd, on=["Stkcd", "Trddt"])
    return df[df["Trddt"] >= START_DATE].sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_mktcap"] = np.log(df["Dsmvtll"].clip(lower=1e-6))

    daily_mean = df.groupby("Trddt")["Dretwd"].transform("mean")
    df["demeaned_ret"] = df["Dretwd"] - daily_mean

    df["turnover_ma"] = df.groupby("Stkcd")["ToverOs"].transform(
        lambda x: x.shift(1).rolling(TURNOVER_WINDOW, min_periods=TURNOVER_WINDOW).mean()
    )
    df["abn_turnover"] = df["ToverOs"] / df["turnover_ma"] - 1

    df["LimitStatus"] = df["LimitStatus"].fillna(0)

    df = df.dropna(subset=["abn_turnover"]).reset_index(drop=True)
    return df[["Stkcd", "Trddt", "demeaned_ret", "log_mktcap", "abn_turnover", "LimitStatus"]]


def main():
    df = load_and_merge()
    df = compute_features(df)
    out = DATA_DIR / "daily_features.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out}: {df.shape}")


if __name__ == "__main__":
    main()
