"""
03c_nonlinear_clustering.py — evidence only. Five independent checks on
whether the note-embedding space has structure beyond what k=15 KMeans
imposes: deepened hierarchical clustering, UMAP+HDBSCAN, a skip-gram
embedding, a self-organising map, and graph community detection.

03_taxonomy.py's k=15 KMeans remains the production taxonomy regardless
of what this script finds. 03 is not imported, read, or modified.

Reuses 03's 50-d PPMI+SVD note embedding from the cache written by
03b_cluster_selection.py (data/interim/03b_note_embedding.npz -- same
method/seed as 03, so this is the identical embedding, not a refit) and
compares every method below against the production KMeans@15 labels
(outputs/taxonomy_map.csv).

Outputs:
    outputs/figures/i_umap_note_map.png
    outputs/figures/j_som_umatrix.png
    outputs/hierarchical_k5_superfamilies.csv
Run standalone: python src/03c_nonlinear_clustering.py
(inputs: data/interim/03b_note_embedding.npz, outputs/taxonomy_map.csv,
outputs/cluster_names_final.csv, data/interim/01_products.parquet,
01_notes_long.parquet, outputs/note_normalisation.csv)
"""
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import umap
from gensim.models import Word2Vec
from minisom import MiniSom
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
SEED = 42
K_PRODUCTION = 15
HIER_LINKAGES = ["ward", "average", "complete", "single"]
HIER_CUTS = [5, 8, 15, 25]
UMAP_NEIGHBOR_SETTINGS = [15, 30]
HDBSCAN_MIN_CLUSTER_SIZE = 10
SOM_GRID = 6
PPMI_EDGE_THRESHOLD = 1.0  # natural-log PPMI >= 1 <=> co-occur >= e~2.7x chance

PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    """Identical to 02/03/03b/08/10's basic_normalize."""
    s = raw.lower().strip()
    if PAREN_RE.search(s):
        s = PAREN_RE.sub("", s)
    return WS_RE.sub(" ", s.replace("-", " ")).strip()


def load_shared():
    npz = np.load(IN_DIR / "03b_note_embedding.npz", allow_pickle=True)
    vocab = list(npz["vocab"])
    embedding, ppmi = npz["embedding"], npz["ppmi"]

    taxonomy_map = pd.read_csv(OUT_DIR / "taxonomy_map.csv")
    note_to_k15 = dict(zip(taxonomy_map["canonical_note"], taxonomy_map["cluster"]))
    kmeans15 = np.array([note_to_k15[w] for w in vocab])

    names_df = pd.read_csv(OUT_DIR / "cluster_names_final.csv")
    short_name = dict(zip(names_df["#"], names_df["Short name"]))
    return vocab, embedding, ppmi, kmeans15, short_name


def load_product_note_sentences():
    products = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population"])
    pop_ids = set(products.loc[products["in_population"], "id"])
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")
    norm_map = pd.read_csv(OUT_DIR / "note_normalisation.csv")
    keep_map = norm_map[norm_map["canonical"].notna()][["raw_token", "canonical"]].drop_duplicates()

    notes = notes_long[notes_long["product_id"].isin(pop_ids)].copy()
    notes["raw_token"] = notes["note_raw"].map(basic_normalize)
    notes = notes.merge(keep_map, on="raw_token", how="inner")
    prod_notes = notes[["product_id", "canonical"]].drop_duplicates()
    return prod_notes.groupby("product_id")["canonical"].apply(list).tolist()


# === 1. hierarchical, deepened ================================================

def section_hierarchical(vocab, embedding, kmeans15, short_name):
    print("=" * 92)
    print("1. HIERARCHICAL, DEEPENED")
    print("=" * 92)
    dist = pdist(embedding, metric="euclidean")

    cophenetic = {}
    Z_by_method = {}
    for method in HIER_LINKAGES:
        Z = linkage(dist, method=method)
        c, _ = cophenet(Z, dist)
        cophenetic[method] = c
        Z_by_method[method] = Z
    print("cophenetic correlation (preservation of pairwise distances):")
    for method in HIER_LINKAGES:
        print(f"  {method:<10} {cophenetic[method]:.4f}")
    best = max(cophenetic, key=cophenetic.get)
    print(f"best linkage by cophenetic correlation: {best} (r={cophenetic[best]:.4f})")

    # cluster-size distribution per cut -- cophenetic correlation alone can be high
    # for a degenerate chaining tree (one mega-cluster + singleton outliers), which
    # preserves LOCAL distance ordering well without being a useful k-way split.
    # Report size distribution explicitly so that isn't silently mistaken for structure.
    print()
    print(f"ARI between {best}-linkage cuts and KMeans@{K_PRODUCTION}, and cut size distribution:")
    cuts = {}
    degenerate = {}
    for k in HIER_CUTS:
        labels = fcluster(Z_by_method[best], k, criterion="maxclust")
        cuts[k] = labels
        ari = adjusted_rand_score(kmeans15, labels)
        sizes = np.sort(np.bincount(labels)[1:])[::-1]
        largest_pct = 100 * sizes[0] / sizes.sum()
        degenerate[k] = largest_pct >= 80
        flag = "  <- DEGENERATE (chaining, not a k-way split)" if degenerate[k] else ""
        print(f"  k={k:<3} ARI={ari:.4f}  largest cluster={largest_pct:.1f}% of notes, "
              f"sizes(top6)={[int(x) for x in sizes[:6]]}{flag}")
    ari_k15 = adjusted_rand_score(kmeans15, cuts[K_PRODUCTION])

    # contrast: ward's size distribution, for context (ward loses on cophenetic
    # correlation but is the only linkage that produces balanced clusters at all --
    # see the printed cophenetic table above; ward's own cuts are not otherwise used).
    ward_k5 = fcluster(Z_by_method["ward"], 5, criterion="maxclust")
    ward_sizes = np.sort(np.bincount(ward_k5)[1:])[::-1]
    print(f"  (for contrast, ward's k=5 cut: largest cluster={100*ward_sizes[0]/ward_sizes.sum():.1f}%, "
          f"sizes={[int(x) for x in ward_sizes]} -- ward has the worst cophenetic correlation but "
          f"the only balanced partition of the four linkages)")

    if degenerate[5]:
        print()
        print(f"NESTING CHECK SKIPPED: {best}-linkage's k=5 cut is degenerate (one cluster holds "
              f"{100 * np.sort(np.bincount(cuts[5])[1:])[::-1][0] / len(vocab):.0f}% of all notes) "
              f"-- there is no real 5-way split to check families against, so no 'nested "
              f"super-family' claim is made. Saving the cut anyway, flagged as degenerate, for "
              f"the record.")

    df = pd.DataFrame({"note": vocab, "family_cluster": kmeans15, "super5": cuts[5]})
    rows = []
    for fc, grp in df.groupby("family_cluster"):
        dominant = grp["super5"].mode().iloc[0]
        purity = (grp["super5"] == dominant).mean()
        rows.append({"family_cluster": fc, "family_name": short_name.get(fc, fc),
                      "super5_group": dominant, "purity": purity, "n_notes": len(grp)})
    nesting = pd.DataFrame(rows)
    nesting["degenerate_cut"] = degenerate[5]
    if not degenerate[5]:
        weighted_purity = (nesting["purity"] * nesting["n_notes"]).sum() / nesting["n_notes"].sum()
        print()
        print(f"nesting check: does each of the 15 production families sit mostly inside one "
              f"{best}-linkage k=5 super-group?")
        print(f"  size-weighted purity: {weighted_purity:.3f} (1.0 = perfectly nested)")
    nesting = nesting.sort_values(["super5_group", "purity"], ascending=[True, False])
    print(nesting.to_string(index=False))
    nesting.to_csv(OUT_DIR / "hierarchical_k5_superfamilies.csv", index=False)
    print(f"saved -> outputs/hierarchical_k5_superfamilies.csv "
          f"({'flagged degenerate' if degenerate[5] else 'coherent cut'})")

    return {
        "method": f"hierarchical_{best}_k{K_PRODUCTION}", "n_clusters": K_PRODUCTION,
        "pct_noise": 0.0, "ari_vs_kmeans15": ari_k15,
        "quality_name": f"cophenetic_corr({best})", "quality_value": cophenetic[best],
    }


# === 2. nonlinear manifold: UMAP -> HDBSCAN ====================================

def section_umap_hdbscan(vocab, embedding, kmeans15, short_name):
    print()
    print("=" * 92)
    print("2. NONLINEAR MANIFOLD: UMAP -> HDBSCAN")
    print("=" * 92)
    rows_out = []
    umap_2d = {}
    for nn in UMAP_NEIGHBOR_SETTINGS:
        reducer = umap.UMAP(n_neighbors=nn, min_dist=0.0, n_components=2, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            emb2d = reducer.fit_transform(embedding)
        umap_2d[nn] = emb2d

        hdb = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, copy=True)
        labels = hdb.fit_predict(emb2d)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        pct_noise = float((labels == -1).mean())
        ari = adjusted_rand_score(kmeans15, labels)
        mask = labels != -1
        sil = (silhouette_score(emb2d[mask], labels[mask])
               if mask.sum() > 1 and len(set(labels[mask])) > 1 else np.nan)

        print(f"n_neighbors={nn}: n_clusters={n_clusters}, pct_noise={pct_noise:.1%}, "
              f"ARI vs KMeans@{K_PRODUCTION}={ari:.4f}, silhouette(non-noise)={sil:.4f}")
        rows_out.append({
            "method": f"umap_hdbscan_nn{nn}", "n_clusters": n_clusters, "pct_noise": pct_noise,
            "ari_vs_kmeans15": ari, "quality_name": "silhouette(non-noise)", "quality_value": sil,
        })

    if all(r["n_clusters"] <= 3 or r["ari_vs_kmeans15"] < 0.05 for r in rows_out):
        print("HEADLINE: UMAP->HDBSCAN still finds ~no cluster structure worth the name -- the "
              "manifold-friendly pipeline that usually recovers what raw HDBSCAN misses finds "
              "essentially nothing here either (few clusters, ~chance-level agreement with the "
              "production taxonomy).")

    # figure: 2-D UMAP scatter (n_neighbors=15), coloured by KMeans@15
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(set(kmeans15)):
        mask = kmeans15 == c
        ax.scatter(umap_2d[15][mask, 0], umap_2d[15][mask, 1], s=22, color=cmap(int(c) % 20),
                   label=short_name.get(c, str(c)), alpha=0.85)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP (n_neighbors=15, min_dist=0.0) projection of the 50-d note embedding\n"
                 f"coloured by production KMeans@{K_PRODUCTION} label")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="family", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "i_umap_note_map.png", dpi=200)
    plt.close(fig)
    print("saved -> outputs/figures/i_umap_note_map.png")

    return rows_out


# === 3. neural embedding: skip-gram Word2Vec ===================================

def section_skipgram(vocab, kmeans15, sentences):
    print()
    print("=" * 92)
    print("3. NEURAL EMBEDDING: skip-gram (gensim Word2Vec, sg=1)")
    print("=" * 92)
    model = Word2Vec(sentences=sentences, sg=1, vector_size=50, window=999, min_count=30,
                      seed=SEED, epochs=30, workers=1)
    sg_vocab = set(model.wv.index_to_key)
    print(f"skip-gram vocabulary: {len(sg_vocab):,} notes (PPMI+SVD vocabulary: {len(vocab):,})")

    note_to_k15 = dict(zip(vocab, kmeans15))
    common = [w for w in vocab if w in sg_vocab]
    dropped = len(vocab) - len(common)
    if dropped:
        print(f"  {dropped} notes in the PPMI+SVD vocab did not clear min_count=30 in skip-gram "
              f"(should be ~0: both use the same population/threshold)")

    sg_matrix = np.stack([model.wv[w] for w in common])
    kmeans15_common = np.array([note_to_k15[w] for w in common])

    km = KMeans(n_clusters=K_PRODUCTION, random_state=SEED, n_init=10)
    sg_labels = km.fit_predict(sg_matrix)
    ari = adjusted_rand_score(kmeans15_common, sg_labels)
    sil = silhouette_score(sg_matrix, sg_labels)
    print(f"KMeans@{K_PRODUCTION} on skip-gram embedding vs production PPMI+SVD partition: "
          f"ARI={ari:.4f}, silhouette={sil:.4f}")
    print("NOTE: skip-gram with negative sampling implicitly factorises a shifted PPMI matrix "
          "(Levy & Goldberg, 2014) -- so HIGH agreement here is the expected result, and confirms "
          "the SVD embedding is not an artefact of one particular factorisation choice, not a "
          "surprising finding of 'two methods agree'.")

    return {"method": "skipgram_kmeans15", "n_clusters": K_PRODUCTION, "pct_noise": 0.0,
            "ari_vs_kmeans15": ari, "quality_name": "silhouette", "quality_value": sil}


# === 4. self-organising map ====================================================

def section_som(vocab, embedding, kmeans15, short_name):
    print()
    print("=" * 92)
    print("4. SELF-ORGANISING MAP")
    print("=" * 92)
    som = MiniSom(SOM_GRID, SOM_GRID, embedding.shape[1], sigma=1.0, learning_rate=0.5,
                  random_seed=SEED)
    som.random_weights_init(embedding)
    som.train(embedding, 5000, random_order=True, verbose=False)

    bmus = [som.winner(x) for x in embedding]
    som_labels = np.array([i * SOM_GRID + j for i, j in bmus])
    n_active_nodes = len(set(som_labels))
    ari = adjusted_rand_score(kmeans15, som_labels)
    qe = som.quantization_error(embedding)
    print(f"{SOM_GRID}x{SOM_GRID} grid ({SOM_GRID * SOM_GRID} nodes), {n_active_nodes} nodes "
          f"received >=1 note, quantization_error={qe:.4f}")
    print(f"ARI between SOM node assignment and KMeans@{K_PRODUCTION}: {ari:.4f}")

    umatrix = som.distance_map()
    print(f"u-matrix range: [{umatrix.min():.3f}, {umatrix.max():.3f}], std={umatrix.std():.3f}")
    if umatrix.max() - umatrix.min() < 0.3:
        print("  u-matrix is nearly flat -- no visible ridges -- further evidence of continuity, "
              "not discrete clusters.")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(umatrix.T, cmap="bone", origin="lower")
    plt.colorbar(im, label="mean distance to neighbouring nodes")
    ax.set_title(f"SOM u-matrix ({SOM_GRID}x{SOM_GRID}, seed {SEED})\n"
                 "ridges would mark cluster boundaries; a flat map means none")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "j_som_umatrix.png", dpi=200)
    plt.close(fig)
    print("saved -> outputs/figures/j_som_umatrix.png")

    return {"method": f"som_{SOM_GRID}x{SOM_GRID}", "n_clusters": n_active_nodes, "pct_noise": 0.0,
            "ari_vs_kmeans15": ari, "quality_name": "quantization_error", "quality_value": qe}


# === 5. graph community detection (Louvain) =====================================

def section_louvain(vocab, ppmi, kmeans15):
    print()
    print("=" * 92)
    print("5. GRAPH COMMUNITY DETECTION (Louvain)")
    print("=" * 92)
    G = nx.Graph()
    G.add_nodes_from(vocab)
    n = len(vocab)
    for i in range(n):
        row = ppmi[i]
        for j in range(i + 1, n):
            if row[j] >= PPMI_EDGE_THRESHOLD:
                G.add_edge(vocab[i], vocab[j], weight=float(row[j]))
    print(f"note graph: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges "
          f"(PPMI >= {PPMI_EDGE_THRESHOLD}), density={nx.density(G):.4f}, "
          f"{sum(1 for _, d in G.degree() if d == 0)} isolated nodes")

    communities = nx.algorithms.community.louvain_communities(G, seed=SEED, weight="weight")
    modularity = nx.algorithms.community.modularity(G, communities, weight="weight")
    node_to_comm = {node: ci for ci, comm in enumerate(communities) for node in comm}
    louvain_labels = np.array([node_to_comm[w] for w in vocab])
    ari = adjusted_rand_score(kmeans15, louvain_labels)

    print(f"n communities: {len(communities)}, modularity: {modularity:.4f}")
    if modularity < 0.3:
        print("  modularity < ~0.3: weak community structure on this graph.")
    print(f"ARI vs KMeans@{K_PRODUCTION}: {ari:.4f}")

    return {"method": "louvain", "n_clusters": len(communities), "pct_noise": 0.0,
            "ari_vs_kmeans15": ari, "quality_name": "modularity", "quality_value": modularity}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vocab, embedding, ppmi, kmeans15, short_name = load_shared()
    print(f"shared embedding: {len(vocab)} notes x {embedding.shape[1]} dims "
          f"(reused from 03b_note_embedding.npz)")
    print(f"production taxonomy: KMeans k={K_PRODUCTION} (outputs/taxonomy_map.csv) -- NOT modified")

    summary_rows = []
    summary_rows.append(section_hierarchical(vocab, embedding, kmeans15, short_name))
    summary_rows.extend(section_umap_hdbscan(vocab, embedding, kmeans15, short_name))

    sentences = load_product_note_sentences()
    summary_rows.append(section_skipgram(vocab, kmeans15, sentences))

    summary_rows.append(section_som(vocab, embedding, kmeans15, short_name))
    summary_rows.append(section_louvain(vocab, ppmi, kmeans15))

    summary = pd.DataFrame(summary_rows)
    summary["pct_noise"] = (summary["pct_noise"] * 100).round(1)
    summary["ari_vs_kmeans15"] = summary["ari_vs_kmeans15"].round(4)
    summary["quality_value"] = summary["quality_value"].round(4)
    summary.to_csv(OUT_DIR / "03c_summary.csv", index=False)

    print()
    print("=" * 92)
    print("SUMMARY (evidence only -- production taxonomy is unchanged: KMeans k=15 in 03_taxonomy.py)")
    print("=" * 92)
    with pd.option_context("display.width", 140, "display.max_columns", 10):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
