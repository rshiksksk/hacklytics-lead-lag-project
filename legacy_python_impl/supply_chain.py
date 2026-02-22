import pandas as pd
import networkx as nx
import pickle
from config import DATA_DIR


def load_supply_chain() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "Supply_Chain_data.csv")
    df["EndDate"] = pd.to_datetime(df["EndDate"], format="%m/%d/%y")
    return df[df["EndDate"].dt.month == 12].copy()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # For identical (Symbol, BusinessSymbol, EndDate, BusinessRelations),
    # prefer consolidated (StateTypeCode=1) over parent-only (2)
    key = ["Symbol", "BusinessSymbol", "EndDate", "BusinessRelations"]
    return (
        df.sort_values("StateTypeCode")
        .drop_duplicates(subset=key, keep="first")
        .reset_index(drop=True)
    )


def get_snapshot_year(trade_date: pd.Timestamp) -> int:
    """Return valid supply chain snapshot year for a given trading date.

    Annual reports announced by Apr 30 of the following year, so:
      dates >= May 1 of year Y  ->  use snapshot Dec 31 of year Y-1
      dates <  May 1 of year Y  ->  use snapshot Dec 31 of year Y-2
    """
    return trade_date.year - 1 if trade_date.month >= 5 else trade_date.year - 2


def _node_id(business_symbol, institution_id) -> str:
    if pd.notna(business_symbol):
        return f"listed_{int(business_symbol)}"
    return f"unlisted_{int(institution_id)}"


def build_graphs(df: pd.DataFrame) -> dict:
    graphs = {}
    for end_date, group in df.groupby("EndDate"):
        G = nx.DiGraph()
        for _, row in group.iterrows():
            src = f"listed_{int(row['Symbol'])}"
            dst = _node_id(row["BusinessSymbol"], row["BusinessInstitutionID"])
            rel = int(row["BusinessRelations"])  # 1=customer, 2=supplier
            rank = int(row["Rank"])

            G.add_edge(src, dst, relation=rel, rank=rank)
            G.add_edge(dst, src, relation=3 - rel, rank=rank)

        graphs[end_date.year] = G
        print(f"  {end_date.year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return graphs


def main():
    df = load_supply_chain()
    df = deduplicate(df)
    print("Building supply chain graphs...")
    graphs = build_graphs(df)
    out = DATA_DIR / "supply_chain_graphs.pkl"
    with open(out, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
