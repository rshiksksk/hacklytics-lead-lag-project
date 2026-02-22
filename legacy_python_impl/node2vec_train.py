import pickle
from node2vec import Node2Vec
from config import DATA_DIR, EMBEDDING_DIM, WALK_LENGTH, NUM_WALKS, N2V_WINDOW, N2V_WORKERS


def train_embeddings(graphs: dict) -> dict:
    embeddings = {}
    for year, G in sorted(graphs.items()):
        print(f"Training Node2Vec for {year} ({G.number_of_nodes()} nodes)...")
        n2v = Node2Vec(
            G,
            dimensions=EMBEDDING_DIM,
            walk_length=WALK_LENGTH,
            num_walks=NUM_WALKS,
            workers=N2V_WORKERS,
            quiet=True,
        )
        model = n2v.fit(window=N2V_WINDOW, min_count=1, batch_words=4)
        embeddings[year] = {node: model.wv[str(node)] for node in G.nodes()}
        print(f"  Done: {len(embeddings[year])} embeddings")
    return embeddings


def main():
    with open(DATA_DIR / "supply_chain_graphs.pkl", "rb") as f:
        graphs = pickle.load(f)

    embeddings = train_embeddings(graphs)

    out = DATA_DIR / "node2vec_embeddings.pkl"
    with open(out, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
