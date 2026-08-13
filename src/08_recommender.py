"""
08_recommender.py — build product vectors and top-50 nearest neighbours for
four fixed representations plus a hybrid, all on the analysis population.

    A. note_svd    mean of a product's canonical note embeddings (03's 50-d
                   PPMI+SVD space, reused from data/interim/03b_note_embedding.npz
                   -- same method/seed as 03_taxonomy.py, cached there), L2-normalised.
    B. note_tfidf  TF-IDF over canonical notes (binary presence x IDF), cosine space.
    C. family      the 15-d fractional family membership vector from 03.
    D. accord      accord strength vector (strengths 0-100 as weights).
    hybrid         z-scored A concatenated with z-scored D.

Neighbours computed with sklearn NearestNeighbors(metric="cosine") which
chunks its pairwise-distance computation internally -- no dense NxN matrix
is ever materialised.

Outputs: data/interim/08_neighbours_{rep}.parquet
    product_id, rank, neighbour_id, similarity
Run standalone: python src/08_recommender.py
(inputs: data/interim/01_products.parquet, 01_notes_long.parquet,
01_accords_long.parquet, 03_product_family.parquet, 03b_note_embedding.npz,
outputs/note_normalisation.csv)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
SEED = 42
TOP_K = 50

PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    """Identical to 02/03/03b's basic_normalize -- needed to rejoin
    01_notes_long against outputs/note_normalisation.csv."""
    s = raw.lower().strip()
    if PAREN_RE.search(s):
        s = PAREN_RE.sub("", s)
    return WS_RE.sub(" ", s.replace("-", " ")).strip()


def l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def top50_neighbours(ids, matrix, rep_name):
    """ids: array of product_id aligned to matrix rows. Returns a long
    dataframe (product_id, rank, neighbour_id, similarity), self excluded."""
    n = matrix.shape[0]
    k = min(TOP_K + 1, n)  # +1 to drop self after the fact
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(matrix)
    dist, idx = nn.kneighbors(matrix)

    rows = []
    for i in range(n):
        pid = ids[i]
        r = 0
        for j, d in zip(idx[i], dist[i]):
            if ids[j] == pid:
                continue  # drop self
            r += 1
            if r > TOP_K:
                break
            rows.append({"product_id": pid, "rank": r, "neighbour_id": ids[j], "similarity": 1.0 - d})
    out = pd.DataFrame(rows)
    out["product_id"] = out["product_id"].astype("Int64")
    out["neighbour_id"] = out["neighbour_id"].astype("Int64")
    out["rank"] = out["rank"].astype("int16")
    print(f"  [{rep_name}] {n:,} products, matrix {matrix.shape}, "
          f"{len(out):,} neighbour rows, mean top-1 similarity "
          f"{out.loc[out['rank'] == 1, 'similarity'].mean():.3f}")
    return out


def main():
    products = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population"])
    pop_ids = set(products.loc[products["in_population"], "id"])
    print(f"analysis population: {len(pop_ids):,}")

    # --- canonical notes per product (population only), same join as 03 -------
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")
    norm_map = pd.read_csv(OUT_DIR / "note_normalisation.csv")
    keep_map = norm_map[norm_map["canonical"].notna()][["raw_token", "canonical"]].drop_duplicates()

    notes = notes_long[notes_long["product_id"].isin(pop_ids)].copy()
    notes["raw_token"] = notes["note_raw"].map(basic_normalize)
    notes = notes.merge(keep_map, on="raw_token", how="inner")
    prod_notes = notes[["product_id", "canonical"]].drop_duplicates()

    note_products = sorted(prod_notes["product_id"].unique())
    n_no_notes = len(pop_ids) - len(note_products)
    print(f"population products with >=1 surviving canonical note: {len(note_products):,} "
          f"({n_no_notes:,} excluded from A/B/C/hybrid: no surviving notes)")

    print()
    print("building representations...")

    # === A. note_svd ==============================================================
    npz = np.load(IN_DIR / "03b_note_embedding.npz", allow_pickle=True)
    vocab, embedding = list(npz["vocab"]), npz["embedding"]
    note_vec = dict(zip(vocab, embedding))

    prod_note_vecs = prod_notes.copy()
    prod_note_vecs["vec"] = prod_note_vecs["canonical"].map(note_vec)
    a_grouped = prod_note_vecs.groupby("product_id")["vec"].apply(lambda s: np.mean(np.stack(s), axis=0))
    a_ids = np.array(a_grouped.index)
    a_matrix = l2_normalize(np.stack(a_grouped.values))
    nbr_a = top50_neighbours(a_ids, a_matrix, "note_svd")
    nbr_a.to_parquet(IN_DIR / "08_neighbours_note_svd.parquet", index=False)

    # === B. note_tfidf =============================================================
    canon_vocab = sorted(prod_notes["canonical"].unique())
    canon_index = {c: i for i, c in enumerate(canon_vocab)}
    prod_index = {p: i for i, p in enumerate(note_products)}
    rows_ = prod_notes["product_id"].map(prod_index).to_numpy()
    cols_ = prod_notes["canonical"].map(canon_index).to_numpy()
    binary = sparse.csr_matrix((np.ones(len(prod_notes)), (rows_, cols_)),
                                shape=(len(note_products), len(canon_vocab)))
    b_matrix = TfidfTransformer(norm="l2").fit_transform(binary)
    b_ids = np.array(note_products)
    nbr_b = top50_neighbours(b_ids, b_matrix, "note_tfidf")
    nbr_b.to_parquet(IN_DIR / "08_neighbours_note_tfidf.parquet", index=False)

    # === C. family ==================================================================
    fam = pd.read_parquet(IN_DIR / "03_product_family.parquet")
    fam_wide = fam.pivot_table(index="product_id", columns="cluster", values="share", fill_value=0.0)
    fam_wide = fam_wide.reindex(note_products, fill_value=0.0)  # same eligible set as A/B
    c_ids = np.array(fam_wide.index)
    c_matrix = fam_wide.to_numpy()
    nbr_c = top50_neighbours(c_ids, c_matrix, "family")
    nbr_c.to_parquet(IN_DIR / "08_neighbours_family.parquet", index=False)

    # === D. accord ===================================================================
    accords_long = pd.read_parquet(IN_DIR / "01_accords_long.parquet")
    acc = accords_long[accords_long["product_id"].isin(pop_ids)].copy()
    acc["accord"] = acc["accord"].str.lower()
    acc["strength"] = acc["strength"].astype("float64")  # nullable Int64 breaks pivot_table's fillna
    acc_wide = acc.pivot_table(index="product_id", columns="accord", values="strength",
                                aggfunc="max", fill_value=0.0)
    nonzero = acc_wide.to_numpy().sum(axis=1) > 0
    n_no_accord = (~nonzero).sum()
    print(f"population products with >=1 nonzero accord: {nonzero.sum():,} "
          f"({n_no_accord:,} excluded from D/hybrid: no accords)")
    acc_wide = acc_wide.loc[nonzero]
    d_ids = np.array(acc_wide.index)
    d_matrix = acc_wide.to_numpy()
    nbr_d = top50_neighbours(d_ids, d_matrix, "accord")
    nbr_d.to_parquet(IN_DIR / "08_neighbours_accord.parquet", index=False)

    # === hybrid: z-scored A concat z-scored D, on the intersection of both =========
    hybrid_ids = sorted(set(a_ids) & set(d_ids))
    a_lookup = {pid: vec for pid, vec in zip(a_ids, a_matrix)}
    d_lookup = {pid: vec for pid, vec in zip(d_ids, d_matrix)}
    a_sub = np.stack([a_lookup[p] for p in hybrid_ids])
    d_sub = np.stack([d_lookup[p] for p in hybrid_ids])

    def zscore_cols(m):
        mean, std = m.mean(axis=0), m.std(axis=0)
        std[std == 0] = 1.0
        return (m - mean) / std

    hybrid_matrix = np.concatenate([zscore_cols(a_sub), zscore_cols(d_sub)], axis=1)
    hybrid_ids = np.array(hybrid_ids)
    print(f"hybrid eligible set (A ∩ D): {len(hybrid_ids):,} products, "
          f"{hybrid_matrix.shape[1]} dims (50 note_svd + {d_sub.shape[1]} accord)")
    nbr_hybrid = top50_neighbours(hybrid_ids, hybrid_matrix, "hybrid")
    nbr_hybrid.to_parquet(IN_DIR / "08_neighbours_hybrid.parquet", index=False)

    # --- diagnostics -------------------------------------------------------------
    print()
    print("=" * 92)
    print("DIAGNOSTICS")
    print("=" * 92)
    for name, df in [("note_svd", nbr_a), ("note_tfidf", nbr_b), ("family", nbr_c),
                      ("accord", nbr_d), ("hybrid", nbr_hybrid)]:
        print(f"  08_neighbours_{name}.parquet  {df.shape}")


if __name__ == "__main__":
    main()
