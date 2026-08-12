"""
03_taxonomy.py — learn an olfactive note taxonomy from co-occurrence.

Canonical-note x canonical-note co-occurrence over in-population products ->
PPMI weighting -> TruncatedSVD (50 dims) -> KMeans (k=15, seed 42; k=12/18
also fit for comparison via silhouette).

Outputs:
    outputs/taxonomy_map.csv        (canonical_note, cluster, svd_x, svd_y, cluster_name, n_products)
    outputs/taxonomy_validation.csv (cluster, cluster_name, accord, lift, ...)
    data/interim/03_product_family.parquet  (product_id, cluster, share)

Run standalone: python src/03_taxonomy.py
(inputs: data/interim/01_products.parquet, 01_notes_long.parquet,
01_accords_long.parquet, outputs/note_normalisation.csv)
"""
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
SEED = 42
SVD_DIMS = 50
K_FINAL = 15
K_CANDIDATES = [12, 15, 18]
TOP_ACCORDS_PER_CLUSTER = 5
MIN_ACCORD_SUPPORT = 100  # accords below this population-wide count (e.g. "wet
# plaster": 1, "tennis ball": 3, "asphault": 10) make lift meaningless -- a
# single product's overlap with a cluster produces lift in the double/triple
# digits. There's a natural break in this corpus around ~100 (cannabis: 100,
# terpenic: 103) vs. the next tier down (champagne: 97, beeswax: 96); accords
# above the break are real signal, below it is noise.

# --- must match src/02_normalise_notes.py's basic_normalize exactly, so that
# raw_token here joins cleanly against outputs/note_normalisation.csv. Not a
# re-run from raw data: 01_notes_long.parquet is the checkpoint; this is pure
# text hygiene applied to it, identical to what produced the CSV we're joining.
PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    s = raw.lower().strip()
    m = PAREN_RE.search(s)
    if m:
        s = PAREN_RE.sub("", s)
    s = s.replace("-", " ")
    s = WS_RE.sub(" ", s).strip()
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population"])
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")
    accords_long = pd.read_parquet(IN_DIR / "01_accords_long.parquet")
    norm_map = pd.read_csv(OUT_DIR / "note_normalisation.csv")

    pop_ids = set(products.loc[products["in_population"], "id"])
    print(f"in_population products: {len(pop_ids):,}")

    # --- map raw notes -> canonical notes, in-population only --------------
    notes = notes_long[notes_long["product_id"].isin(pop_ids)].copy()
    notes["raw_token"] = notes["note_raw"].map(basic_normalize)

    keep_map = norm_map[norm_map["canonical"].notna()][["raw_token", "canonical"]].drop_duplicates()
    notes = notes.merge(keep_map, on="raw_token", how="inner")

    # dedupe: a product's canonical note counts once even if it arrived via
    # multiple tiers/raw variants (e.g. top "Cedar" + base "Cedarwood" -> both
    # canonical "cedar" -- one occurrence, not two, for co-occurrence purposes)
    prod_notes = notes[["product_id", "canonical"]].drop_duplicates()

    n_with_notes = prod_notes["product_id"].nunique()
    n_dropped_all_notes = len(pop_ids) - n_with_notes
    print(f"in-population products with >=1 surviving canonical note: {n_with_notes:,}")
    print(f"in-population products with ZERO surviving canonical notes "
          f"(all notes were DROP_PLACEHOLDER/DROP_RARE): {n_dropped_all_notes:,}")

    vocab = sorted(prod_notes["canonical"].unique())
    vocab_index = {c: i for i, c in enumerate(vocab)}
    V = len(vocab)
    print(f"canonical note vocabulary: {V:,}")

    # --- co-occurrence matrix (symmetric, zero diagonal) --------------------
    co = np.zeros((V, V), dtype=np.float64)
    for _, grp in prod_notes.groupby("product_id")["canonical"]:
        idxs = sorted({vocab_index[c] for c in grp})
        for i, j in combinations(idxs, 2):
            co[i, j] += 1
            co[j, i] += 1

    print(f"co-occurrence matrix: {V}x{V}, {int(co.sum() / 2):,} unordered note-pairs")

    # --- PPMI ----------------------------------------------------------------
    row_sums = co.sum(axis=1)
    total = co.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.outer(row_sums, row_sums) / total
        pmi = np.log(co * total / np.outer(row_sums, row_sums))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)

    # --- TruncatedSVD to 50 dims ---------------------------------------------
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=SEED)
    embedding = svd.fit_transform(ppmi)
    print(f"TruncatedSVD: {V}x{V} PPMI -> {embedding.shape[1]} dims, "
          f"explained variance ratio sum = {svd.explained_variance_ratio_.sum():.3f}")

    # --- KMeans: compare k=12/15/18 via silhouette, use k=15 -----------------
    print()
    print("k comparison (silhouette on 50-dim embedding):")
    labels_by_k = {}
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(embedding)
        sil = silhouette_score(embedding, labels)
        labels_by_k[k] = labels
        marker = "  <- used" if k == K_FINAL else ""
        print(f"  k={k:<3} silhouette={sil:.4f}{marker}")

    labels = labels_by_k[K_FINAL]
    cluster_of = dict(zip(vocab, labels))

    # --- per-canonical-note product counts (recomputed on this population) --
    note_n_products = prod_notes.groupby("canonical")["product_id"].nunique()

    # --- validation: top accords per cluster by lift -------------------------
    # product-level accord presence (ignore strength; lift is about mention
    # rate). Restrict to in-population products with a surviving note (the
    # taxonomy population).
    acc = accords_long[accords_long["product_id"].isin(pop_ids)].copy()
    acc = acc[acc["product_id"].isin(prod_notes["product_id"].unique())]
    acc["accord"] = acc["accord"].str.lower()  # source data has inconsistent
    # case on a few entries (e.g. "Champagne", "Pear") though the controlled
    # vocabulary is still exactly 92 either way -- normalize for display.
    acc_presence = acc[["product_id", "accord"]].drop_duplicates()

    n_family_pop = prod_notes["product_id"].nunique()
    accord_counts = acc_presence.groupby("accord")["product_id"].nunique()
    accord_base_rate = accord_counts / n_family_pop
    eligible_accords = set(accord_counts[accord_counts >= MIN_ACCORD_SUPPORT].index)
    n_accords_excluded = accord_counts.shape[0] - len(eligible_accords)
    print(f"accords with population count >= {MIN_ACCORD_SUPPORT} (eligible for "
          f"validation ranking): {len(eligible_accords)}/{accord_counts.shape[0]} "
          f"({n_accords_excluded} excluded as too rare for a stable lift estimate)")

    # fractional family membership per product (share of its canonical notes
    # in each cluster) -- same table saved as 03_product_family.parquet, used
    # here to weight the validation accord-mention rate per cluster.
    prod_notes["cluster"] = prod_notes["canonical"].map(cluster_of)
    membership = (
        prod_notes.groupby(["product_id", "cluster"]).size().rename("n_in_cluster").reset_index()
    )
    totals = prod_notes.groupby("product_id").size().rename("n_total")
    membership = membership.merge(totals, on="product_id")
    membership["share"] = membership["n_in_cluster"] / membership["n_total"]
    membership = membership[["product_id", "cluster", "share"]]

    acc_membership = acc_presence.merge(membership, on="product_id", how="inner")
    cluster_accord_weight = (
        acc_membership.groupby(["cluster", "accord"])["share"].sum().rename("weight").reset_index()
    )
    cluster_size = membership.groupby("cluster")["share"].sum().rename("cluster_weight")
    cluster_accord_weight = cluster_accord_weight.merge(cluster_size, on="cluster")
    cluster_accord_weight["mention_rate"] = (
        cluster_accord_weight["weight"] / cluster_accord_weight["cluster_weight"]
    )
    cluster_accord_weight["base_rate"] = cluster_accord_weight["accord"].map(accord_base_rate)
    cluster_accord_weight["lift"] = cluster_accord_weight["mention_rate"] / cluster_accord_weight["base_rate"]

    validation_rows = []
    cluster_name = {}
    for c in range(K_FINAL):
        sub = cluster_accord_weight[
            (cluster_accord_weight["cluster"] == c) &
            (cluster_accord_weight["accord"].isin(eligible_accords))
        ].sort_values("lift", ascending=False)
        top = sub.head(TOP_ACCORDS_PER_CLUSTER)
        name = top.iloc[0]["accord"] if len(top) else f"cluster_{c}"
        cluster_name[c] = name
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            validation_rows.append({
                "cluster": c, "cluster_name": name, "rank": rank,
                "accord": row["accord"], "lift": round(row["lift"], 3),
                "mention_rate": round(row["mention_rate"], 4),
                "base_rate": round(row["base_rate"], 4),
            })

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(OUT_DIR / "taxonomy_validation.csv", index=False)

    # --- taxonomy_map.csv -----------------------------------------------------
    # svd_x/svd_y: first two components of the same 50-dim SVD embedding (SVD
    # components are hierarchical/nested, so this is exactly the top-2 SVD
    # projection, not a separate refit) -- persisted here so 07_figures.py can
    # plot the note map from this checkpoint without recomputing the pipeline.
    taxonomy_map = pd.DataFrame({
        "canonical_note": vocab,
        "cluster": labels,
        "svd_x": embedding[:, 0],
        "svd_y": embedding[:, 1],
    })
    taxonomy_map["cluster_name"] = taxonomy_map["cluster"].map(cluster_name)
    taxonomy_map["n_products"] = taxonomy_map["canonical_note"].map(note_n_products).astype(int)
    taxonomy_map = taxonomy_map.sort_values(["cluster", "n_products"], ascending=[True, False])
    taxonomy_map.to_csv(OUT_DIR / "taxonomy_map.csv", index=False)

    # --- 03_product_family.parquet --------------------------------------------
    family_df = membership.copy()
    family_df["cluster_name"] = family_df["cluster"].map(cluster_name)
    family_df["product_id"] = family_df["product_id"].astype("Int64")
    family_df["cluster"] = family_df["cluster"].astype("Int64")
    family_df.to_parquet(IN_DIR / "03_product_family.parquet", index=False)

    # --- diagnostics -----------------------------------------------------------
    print()
    print("=" * 88)
    print("DIAGNOSTICS")
    print("=" * 88)
    print(f"in_population products: {len(pop_ids):,}")
    print(f"  - with >=1 surviving canonical note: {n_with_notes:,}")
    print(f"  - with 0 surviving canonical notes (dropped from taxonomy): {n_dropped_all_notes:,}")
    print(f"canonical note vocabulary clustered: {V:,}")
    print(f"final k = {K_FINAL}, silhouette = {silhouette_score(embedding, labels):.4f}")
    print()
    print("cluster sizes (by distinct canonical notes) and auto-name:")
    sizes = taxonomy_map.groupby(["cluster", "cluster_name"]).size().rename("n_notes").reset_index()
    print(sizes.to_string(index=False))
    print()
    print("output shapes:")
    print(f"  outputs/taxonomy_map.csv          {taxonomy_map.shape}")
    print(f"  outputs/taxonomy_validation.csv   {validation_df.shape}")
    print(f"  03_product_family.parquet         {family_df.shape}")
    print()
    print("taxonomy_validation.csv (top accords by lift per cluster):")
    print(validation_df.to_string(index=False))


if __name__ == "__main__":
    main()
