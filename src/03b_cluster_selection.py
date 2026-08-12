"""
03b_cluster_selection.py — evidence for the k/algorithm choice in
03_taxonomy.py. Does NOT change the main pipeline's k (15) or algorithm
(KMeans). This script only produces diagnostics; 03_taxonomy.py is untouched.

Reuses 03's co-occurrence -> PPMI -> TruncatedSVD(50) note embedding. 03
doesn't persist the full 50-d embedding (only its first 2 components, for the
note-map figure), so this script rebuilds it with the identical method/seed
and caches the result to data/interim/03b_note_embedding.npz -- subsequent
runs load the cache instead of recomputing.

    1. k-sweep (2..30): inertia, silhouette, Calinski-Harabasz, Davies-Bouldin,
       cluster-size stats -> outputs/figures/d_cluster_selection.png
       + kneed.KneeLocator on the inertia curve.
    2. Bootstrap stability (k in {8,10,12,15,18,20,25}): 20 product resamples,
       full co-occurrence->PPMI->SVD->KMeans rebuild each, pairwise ARI
       between runs -> outputs/figures/e_cluster_stability.png
    3. External validity: AMI between clustering and each note's hard accord
       label (highest-lift eligible accord) -> outputs/figures/g_ami_vs_k.png
    4. Alternative algorithms at k=15: Agglomerative (ward; cosine-average,
       with dendrogram), HDBSCAN (min_cluster_size 5/10/20), Gaussian Mixture
       (full/diag, + BIC-vs-k), Spectral (on the co-occurrence graph).
       Cross-tabulated against KMeans@15 via ARI
       -> outputs/cluster_algorithm_comparison.csv
    5. Combined summary table (printed).

Run standalone: python src/03b_cluster_selection.py
(inputs: data/interim/01_products.parquet, 01_notes_long.parquet,
01_accords_long.parquet, outputs/note_normalisation.csv)
"""
import re
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import (AgglomerativeClustering, HDBSCAN, KMeans, SpectralClustering)
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                              calinski_harabasz_score, davies_bouldin_score,
                              silhouette_score)
from sklearn.mixture import GaussianMixture

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
CACHE_PATH = IN_DIR / "03b_note_embedding.npz"

SEED = 42
SVD_DIMS = 50
CHOSEN_K = 15  # the main pipeline's k -- not changed by this script
K_SWEEP = list(range(2, 31))
STABILITY_KS = [8, 10, 12, 15, 18, 20, 25]
N_BOOTSTRAP = 20
MIN_ACCORD_SUPPORT = 100  # matches 03_taxonomy.py's threshold, duplicated here
# since this script must not import from/modify 03_taxonomy.py.

PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    """Identical to 02_normalise_notes.py / 03_taxonomy.py's basic_normalize."""
    s = raw.lower().strip()
    m = PAREN_RE.search(s)
    if m:
        s = PAREN_RE.sub("", s)
    s = s.replace("-", " ")
    s = WS_RE.sub(" ", s).strip()
    return s


# --- data loading / embedding (cached) --------------------------------------

def load_population_notes():
    products = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population"])
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")
    norm_map = pd.read_csv(OUT_DIR / "note_normalisation.csv")

    pop_ids = set(products.loc[products["in_population"], "id"])
    notes = notes_long[notes_long["product_id"].isin(pop_ids)].copy()
    notes["raw_token"] = notes["note_raw"].map(basic_normalize)
    keep_map = norm_map[norm_map["canonical"].notna()][["raw_token", "canonical"]].drop_duplicates()
    notes = notes.merge(keep_map, on="raw_token", how="inner")
    prod_notes = notes[["product_id", "canonical"]].drop_duplicates()
    return pop_ids, prod_notes


def build_indicator(prod_notes, vocab_index):
    """Sparse (n_products x n_notes) binary indicator, product order fixed by
    first appearance -- returned alongside the product id list so callers can
    build matching indicators for other product-level signals (e.g. accords)."""
    products = prod_notes["product_id"].unique()
    prod_index = {p: i for i, p in enumerate(products)}
    rows = prod_notes["product_id"].map(prod_index).to_numpy()
    cols = prod_notes["canonical"].map(vocab_index).to_numpy()
    data = np.ones(len(prod_notes))
    X = sparse.csr_matrix((data, (rows, cols)), shape=(len(products), len(vocab_index)))
    X.data[:] = 1.0
    return X, products, prod_index


def cooccurrence_from_indicator(X, weights=None):
    """(notes x notes) co-occurrence, optionally product-weighted (bootstrap)."""
    if weights is not None:
        X = X.multiply(weights[:, None]).tocsr()
        co = (X.T @ X).toarray()
    else:
        co = (X.T @ X).toarray()
    np.fill_diagonal(co, 0.0)
    return co


def ppmi_from_cooccurrence(co):
    row_sums = co.sum(axis=1)
    total = co.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(co * total / np.outer(row_sums, row_sums))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)
    return ppmi


def svd_embedding(ppmi, seed=SEED):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=seed)
    return svd.fit_transform(ppmi)


def get_embedding(prod_notes):
    vocab = sorted(prod_notes["canonical"].unique())
    if CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if list(cached["vocab"]) == vocab:
            print(f"loaded cached embedding: {CACHE_PATH} (vocab matches, {len(vocab)} notes)")
            return vocab, cached["embedding"], cached["co"], cached["ppmi"]
        print("cached embedding vocab mismatch -- recomputing")

    print("no valid cache -- recomputing co-occurrence -> PPMI -> SVD (same method as 03_taxonomy.py)")
    vocab_index = {c: i for i, c in enumerate(vocab)}
    X, _, _ = build_indicator(prod_notes, vocab_index)
    co = cooccurrence_from_indicator(X)
    ppmi = ppmi_from_cooccurrence(co)
    embedding = svd_embedding(ppmi)
    np.savez(CACHE_PATH, vocab=np.array(vocab, dtype=object), embedding=embedding, co=co, ppmi=ppmi)
    print(f"cached embedding -> {CACHE_PATH}")
    return vocab, embedding, co, ppmi


# --- part 1: k-sweep ----------------------------------------------------------

def cluster_size_stats(labels, k):
    counts = np.bincount(labels, minlength=k)
    return counts.mean(), counts.max(), int((counts == 1).sum())


def run_k_sweep(embedding):
    rows = []
    labels_by_k = {}
    for k in K_SWEEP:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(embedding)
        labels_by_k[k] = labels
        mean_size, max_size, n_singleton = cluster_size_stats(labels, k)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(embedding, labels),
            "calinski_harabasz": calinski_harabasz_score(embedding, labels),
            "davies_bouldin": davies_bouldin_score(embedding, labels),
            "mean_cluster_size": mean_size,
            "max_cluster_size": max_size,
            "n_singleton_clusters": n_singleton,
        })
    sweep_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(sweep_df["k"], sweep_df["inertia"], marker="o", markersize=3)
    axes[0, 0].axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1, label=f"chosen k={CHOSEN_K}")
    axes[0, 0].set_title("Inertia (elbow)")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(sweep_df["k"], sweep_df["silhouette"], marker="o", markersize=3, color="tab:orange")
    axes[0, 1].axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Silhouette (higher better)")
    axes[0, 1].set_xlabel("k")

    axes[1, 0].plot(sweep_df["k"], sweep_df["calinski_harabasz"], marker="o", markersize=3, color="tab:green")
    axes[1, 0].axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Calinski-Harabasz (higher better)")
    axes[1, 0].set_xlabel("k")

    axes[1, 1].plot(sweep_df["k"], sweep_df["davies_bouldin"], marker="o", markersize=3, color="tab:purple")
    axes[1, 1].axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Davies-Bouldin (lower better)")
    axes[1, 1].set_xlabel("k")

    fig.suptitle("Cluster-count diagnostics, k=2..30 (KMeans, seed 42, n_init=10)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d_cluster_selection.png", dpi=200)
    plt.close(fig)

    knee = KneeLocator(sweep_df["k"], sweep_df["inertia"], curve="convex", direction="decreasing")
    if knee.knee is None:
        print("kneed.KneeLocator: NO KNEE FOUND on the inertia curve -- not forcing one.")
    else:
        print(f"kneed.KneeLocator: knee at k={knee.knee} (inertia={knee.knee_y:.2f})")

    return sweep_df, labels_by_k


# --- part 2: bootstrap stability -----------------------------------------------

def run_stability(prod_notes):
    vocab = sorted(prod_notes["canonical"].unique())
    vocab_index = {c: i for i, c in enumerate(vocab)}
    X, products, _ = build_indicator(prod_notes, vocab_index)
    n_products = len(products)

    rng = np.random.default_rng(SEED)
    labels_by_k = {k: [] for k in STABILITY_KS}

    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_products, size=n_products)
        weights = np.bincount(idx, minlength=n_products).astype(np.float64)
        co = cooccurrence_from_indicator(X, weights=weights)
        ppmi = ppmi_from_cooccurrence(co)
        embedding = svd_embedding(ppmi, seed=SEED)
        for k in STABILITY_KS:
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels_by_k[k].append(km.fit_predict(embedding))

    rows = []
    for k in STABILITY_KS:
        runs = labels_by_k[k]
        aris = [adjusted_rand_score(runs[i], runs[j]) for i, j in combinations(range(N_BOOTSTRAP), 2)]
        rows.append({"k": k, "mean_ari": float(np.mean(aris)), "std_ari": float(np.std(aris))})
    stability_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(stability_df["k"], stability_df["mean_ari"], yerr=stability_df["std_ari"],
                marker="o", capsize=4, color="tab:blue")
    ax.axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1, label=f"chosen k={CHOSEN_K}")
    ax.set_xlabel("k")
    ax.set_ylabel("mean pairwise ARI across 20 bootstrap resamples (+/- std)")
    ax.set_title(f"Bootstrap stability ({N_BOOTSTRAP} product resamples, full "
                 "co-occurrence->PPMI->SVD->KMeans rebuild per resample)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e_cluster_stability.png", dpi=200)
    plt.close(fig)

    return stability_df


# --- part 3: external validity (AMI vs hard accord labels) --------------------

def hard_accord_labels(prod_notes, vocab):
    products_masked = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population"])
    pop_ids_with_notes = set(prod_notes["product_id"].unique())
    accords_long = pd.read_parquet(IN_DIR / "01_accords_long.parquet")
    acc = accords_long[accords_long["product_id"].isin(pop_ids_with_notes)].copy()
    acc["accord"] = acc["accord"].str.lower()
    acc_presence = acc[["product_id", "accord"]].drop_duplicates()

    n_pop = len(pop_ids_with_notes)
    accord_counts = acc_presence.groupby("accord")["product_id"].nunique()
    eligible = sorted(accord_counts[accord_counts >= MIN_ACCORD_SUPPORT].index)
    base_rate = (accord_counts[eligible] / n_pop).to_numpy()

    vocab_index = {c: i for i, c in enumerate(vocab)}
    X, products, prod_index = build_indicator(prod_notes, vocab_index)  # products x notes

    accord_index = {a: i for i, a in enumerate(eligible)}
    rows = acc_presence["product_id"].map(prod_index)
    keep = rows.notna() & acc_presence["accord"].isin(eligible)
    rows = rows[keep].astype(int).to_numpy()
    cols = acc_presence.loc[keep, "accord"].map(accord_index).to_numpy()
    Y = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(products), len(eligible)))

    note_accord_counts = (X.T @ Y).toarray()  # notes x eligible_accords
    note_totals = np.asarray(X.sum(axis=0)).ravel()
    mention_rate = note_accord_counts / note_totals[:, None]
    lift = mention_rate / base_rate[None, :]
    hard_label_idx = lift.argmax(axis=1)
    hard_label = np.array(eligible)[hard_label_idx]
    print(f"hard accord labels: {len(eligible)} eligible accords (population count >= "
          f"{MIN_ACCORD_SUPPORT}), {pd.Series(hard_label).nunique()} distinct labels used across {len(vocab)} notes")
    return hard_label


def run_external_validity(labels_by_k, hard_label):
    rows = []
    for k in STABILITY_KS:
        ami = adjusted_mutual_info_score(labels_by_k[k], hard_label)
        rows.append({"k": k, "ami_vs_accords": ami})
    ext_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ext_df["k"], ext_df["ami_vs_accords"], marker="o", color="tab:brown")
    ax.axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1, label=f"chosen k={CHOSEN_K}")
    ax.set_xlabel("k")
    ax.set_ylabel("Adjusted Mutual Information vs. hard accord label")
    ax.set_title("External validity: clustering vs. each note's highest-lift eligible accord")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "g_ami_vs_k.png", dpi=200)
    plt.close(fig)
    return ext_df


# --- part 4: alternative algorithms at k=15 ------------------------------------

def run_alternatives(embedding, co, kmeans15_labels, hard_label):
    results = []
    labels_lookup = {}

    # Agglomerative, ward/euclidean
    agg_ward = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage="ward")
    lab = agg_ward.fit_predict(embedding)
    labels_lookup["agglomerative_ward"] = lab
    results.append(("agglomerative_ward", "euclidean/ward", CHOSEN_K, lab))

    # Agglomerative, average/cosine
    agg_cos = AgglomerativeClustering(n_clusters=CHOSEN_K, linkage="average", metric="cosine")
    lab = agg_cos.fit_predict(embedding)
    labels_lookup["agglomerative_cosine_average"] = lab
    results.append(("agglomerative_cosine_average", "cosine/average", CHOSEN_K, lab))

    # dendrogram: both linkages, side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    Z_ward = linkage(embedding, method="ward")
    dendrogram(Z_ward, ax=axes[0], truncate_mode="lastp", p=30, show_contracted=True, no_labels=True)
    axes[0].set_title("Ward / euclidean")
    axes[0].set_xlabel("notes (merged leaves)")
    axes[0].set_ylabel("distance")

    cos_dist = pdist(embedding, metric="cosine")
    Z_cos = linkage(cos_dist, method="average")
    dendrogram(Z_cos, ax=axes[1], truncate_mode="lastp", p=30, show_contracted=True, no_labels=True)
    axes[1].set_title("Average / cosine")
    axes[1].set_xlabel("notes (merged leaves)")
    fig.suptitle(f"Dendrograms on the 50-d note embedding (truncated to last 30 merges), "
                 f"reference n_clusters={CHOSEN_K}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f_dendrogram.png", dpi=200)
    plt.close(fig)

    # HDBSCAN, min_cluster_size 5/10/20
    for mcs in (5, 10, 20):
        hdb = HDBSCAN(min_cluster_size=mcs)
        lab = hdb.fit_predict(embedding)
        name = f"hdbscan_mcs{mcs}"
        labels_lookup[name] = lab
        results.append((name, f"min_cluster_size={mcs}", len(set(lab)) - (1 if -1 in lab else 0), lab))

    # Gaussian Mixture, full & diag, at chosen k
    for cov in ("full", "diag"):
        gmm = GaussianMixture(n_components=CHOSEN_K, covariance_type=cov, random_state=SEED)
        lab = gmm.fit_predict(embedding)
        name = f"gmm_{cov}"
        labels_lookup[name] = lab
        results.append((name, f"covariance={cov}", CHOSEN_K, lab))

    # Spectral clustering on the note co-occurrence graph (raw counts as affinity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = SpectralClustering(n_clusters=CHOSEN_K, affinity="precomputed", random_state=SEED)
        lab = spec.fit_predict(co)
    labels_lookup["spectral_cooccurrence"] = lab
    results.append(("spectral_cooccurrence", "affinity=co-occurrence", CHOSEN_K, lab))

    # --- BIC vs k for GMM (full & diag), separate diagnostic figure ------------
    bic_rows = []
    for cov in ("full", "diag"):
        for k in K_SWEEP:
            gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=SEED)
            gmm.fit(embedding)
            bic_rows.append({"k": k, "covariance": cov, "bic": gmm.bic(embedding)})
    bic_df = pd.DataFrame(bic_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cov, grp in bic_df.groupby("covariance"):
        ax.plot(grp["k"], grp["bic"], marker="o", markersize=3, label=cov)
    ax.axvline(CHOSEN_K, color="red", linestyle="--", linewidth=1, label=f"chosen k={CHOSEN_K}")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC (lower better)")
    ax.set_title("Gaussian Mixture BIC vs k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "h_gmm_bic.png", dpi=200)
    plt.close(fig)

    # --- cross-tabulate each alternative against KMeans@15 ----------------------
    comparison_rows = []
    for name, param, n_clusters_found, lab in results:
        ari = adjusted_rand_score(kmeans15_labels, lab)
        ami_accords = adjusted_mutual_info_score(lab, hard_label)
        has_noise = -1 in lab
        pct_noise = float((lab == -1).mean()) if has_noise else 0.0
        lab_for_metrics = lab
        mask = lab != -1
        if has_noise and mask.sum() > 1 and len(set(lab[mask])) > 1:
            sil = silhouette_score(embedding[mask], lab[mask])
        elif not has_noise and len(set(lab)) > 1:
            sil = silhouette_score(embedding, lab)
        else:
            sil = np.nan
        comparison_rows.append({
            "algorithm": name, "param": param, "n_clusters_found": n_clusters_found,
            "ari_vs_kmeans15": ari, "ami_vs_accords": ami_accords,
            "pct_noise": pct_noise, "silhouette": sil,
        })
        print(f"\n{name} ({param}) vs KMeans@{CHOSEN_K} -- ARI={ari:.3f}, "
              f"n_clusters_found={n_clusters_found}, pct_noise={pct_noise:.1%}")
        print("confusion-style agreement (rows=KMeans@15, cols=" + name + "):")
        ct = pd.crosstab(pd.Series(kmeans15_labels, name="kmeans15"), pd.Series(lab, name=name))
        print(ct.to_string())

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "cluster_algorithm_comparison.csv", index=False)
    return comparison_df, labels_lookup, bic_df


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pop_ids, prod_notes = load_population_notes()
    vocab, embedding, co, ppmi = get_embedding(prod_notes)
    print(f"note vocabulary: {len(vocab)}; embedding shape: {embedding.shape}")

    print()
    print("=" * 100)
    print("PART 1: k-SWEEP (2..30)")
    print("=" * 100)
    sweep_df, sweep_labels_by_k = run_k_sweep(embedding)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(sweep_df.to_string(index=False))

    print()
    print("=" * 100)
    print(f"PART 2: BOOTSTRAP STABILITY ({N_BOOTSTRAP} resamples, k in {STABILITY_KS})")
    print("=" * 100)
    stability_df = run_stability(prod_notes)
    print(stability_df.to_string(index=False))

    print()
    print("=" * 100)
    print("PART 3: EXTERNAL VALIDITY (AMI vs hard accord label)")
    print("=" * 100)
    hard_label = hard_accord_labels(prod_notes, vocab)
    labels_by_k_stability = {k: sweep_labels_by_k[k] for k in STABILITY_KS}
    ext_df = run_external_validity(labels_by_k_stability, hard_label)
    print(ext_df.to_string(index=False))

    print()
    print("=" * 100)
    print(f"PART 4: ALTERNATIVE ALGORITHMS at k={CHOSEN_K}")
    print("=" * 100)
    kmeans15_labels = sweep_labels_by_k[CHOSEN_K]
    comparison_df, labels_lookup, bic_df = run_alternatives(embedding, co, kmeans15_labels, hard_label)

    print()
    print("=" * 100)
    print("PART 5: SUMMARY TABLE")
    print("=" * 100)
    summary_rows = []
    k15_row = sweep_df[sweep_df["k"] == CHOSEN_K].iloc[0]
    k15_stability = stability_df[stability_df["k"] == CHOSEN_K]
    k15_ext = ext_df[ext_df["k"] == CHOSEN_K]
    summary_rows.append({
        "algorithm": "kmeans", "k": CHOSEN_K,
        "silhouette": k15_row["silhouette"], "calinski_harabasz": k15_row["calinski_harabasz"],
        "davies_bouldin": k15_row["davies_bouldin"],
        "mean_bootstrap_ari": k15_stability["mean_ari"].iloc[0] if len(k15_stability) else np.nan,
        "ami_vs_accords": k15_ext["ami_vs_accords"].iloc[0] if len(k15_ext) else np.nan,
        "largest_cluster_pct": k15_row["max_cluster_size"] / len(vocab),
        "n_noise_points": 0,
    })
    for _, row in comparison_df.iterrows():
        lab = labels_lookup[row["algorithm"]]
        mask = lab != -1
        largest_pct = (np.bincount(lab[mask]).max() / mask.sum()) if mask.sum() else np.nan
        sil = row["silhouette"]
        ch = calinski_harabasz_score(embedding[mask], lab[mask]) if mask.sum() > 1 and len(set(lab[mask])) > 1 else np.nan
        db = davies_bouldin_score(embedding[mask], lab[mask]) if mask.sum() > 1 and len(set(lab[mask])) > 1 else np.nan
        stab_row = stability_df[stability_df["k"] == row["n_clusters_found"]]
        summary_rows.append({
            "algorithm": f"{row['algorithm']} ({row['param']})", "k": row["n_clusters_found"],
            "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db,
            "mean_bootstrap_ari": np.nan,  # bootstrap stability run only for KMeans (part 2 scope)
            "ami_vs_accords": row["ami_vs_accords"],
            "largest_cluster_pct": largest_pct,
            "n_noise_points": int((lab == -1).sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary_df.to_string(index=False))

    print()
    print("=" * 100)
    print("DIAGNOSTICS")
    print("=" * 100)
    for p in ["d_cluster_selection.png", "e_cluster_stability.png", "g_ami_vs_k.png",
              "f_dendrogram.png", "h_gmm_bic.png"]:
        fp = FIG_DIR / p
        print(f"  outputs/figures/{p}  ({fp.stat().st_size / 1024:.0f} KB)")
    print(f"  outputs/cluster_algorithm_comparison.csv  {comparison_df.shape}")
    print()
    print(f"main pipeline unchanged: 03_taxonomy.py still uses k={CHOSEN_K}, KMeans. "
          "This script produced evidence only -- no decision made here.")


if __name__ == "__main__":
    main()
