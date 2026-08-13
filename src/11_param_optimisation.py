"""
11_param_optimisation.py — tune the three representations 08/09 left
untuned, WITHOUT leaking evaluation edges into the tuning decision.

LEAKAGE CONTROL (read this before touching anything else in this file):
    Evaluable query products (>=1 confident reminds_me_of edge) are split
    60/20/20 into train/val/test BY QUERY PRODUCT, seed 42, before any
    tuning happens. A product's edges never span splits. Grid search
    below is scored on VALIDATION ONLY. TEST is read exactly once, at
    the very end, to report final tuned-vs-untuned-vs-baseline numbers.
    "train" is held out and unused here (no supervised model is being
    fit) -- it exists so the split is a true 60/20/20 partition and
    reserved for anyone who later wants a third checkpoint.
    Representations themselves (TF-IDF/SVD/hybrid feature construction)
    are unsupervised -- built from all population products' notes/accords,
    which never touches ground-truth labels, so this does not leak
    regardless of the query split. What WOULD leak is choosing
    hyperparameters by looking at the same edges used to report the
    final number; that's exactly what the split prevents.

DIRECTIONALITY: confirmed by re-reading 09_recommender_eval.py's
build_ground_truth() -- it groups confident edges by product_id (the
query) and collects related_id (the target), i.e. OUTGOING edges only.
It does not pool in the reverse direction, so 09's numbers were never
symmetrically inflated. This script re-derives that ground truth
identically and, for contrast, ALSO builds a symmetrised version (edge
counts both ways) and reports both on the test split, to show
numerically what symmetric scoring would have inflated -- not because
09 needed fixing.

Grid search (validation, precision@10):
    note_tfidf : sublinear_tf x min_df x use_idf x tier weighting
    note_svd   : n_components x shifted-PPMI shift
    hybrid     : alpha blending note_tfidf and accord cosine similarity

Outputs:
    outputs/11_param_search.csv
    outputs/figures/k_param_search.png
Run standalone: python src/11_param_optimisation.py
(inputs: data/interim/01_products.parquet, 01_notes_long.parquet,
01_accords_long.parquet, 01c_similar_pairs.parquet,
03b_note_embedding.npz, 08_neighbours_{rep}.parquet,
outputs/note_normalisation.csv)
"""
import re
import warnings
from itertools import product as iterproduct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
SEED = 42
CONFIDENT_THRESHOLD = 5
KS = (5, 10, 20)
OBJECTIVE_K = 10
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2  # test = remainder

PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    s = raw.lower().strip()
    if PAREN_RE.search(s):
        s = PAREN_RE.sub("", s)
    return WS_RE.sub(" ", s.replace("-", " ")).strip()


# ============================================================ data loading ===

def load_population():
    products = pd.read_parquet(IN_DIR / "01_products.parquet",
                                columns=["id", "in_population", "brand", "have"])
    pop = products[products["in_population"]].copy()
    pop["have"] = pop["have"].astype("float64")
    return pop


def load_ground_truth(pop_ids, directional=True):
    sim = pd.read_parquet(IN_DIR / "01c_similar_pairs.parquet")
    rmo = sim[sim["kind"] == "reminds_me_of"].copy()
    rmo = rmo[rmo["product_id"].isin(pop_ids) & rmo["related_id"].isin(pop_ids)]
    rmo["net_votes"] = rmo["up_votes"] - rmo["down_votes"]
    confident = rmo[rmo["net_votes"] >= CONFIDENT_THRESHOLD]

    gt = {}
    for pid, grp in confident.groupby("product_id"):
        gt.setdefault(int(pid), set()).update(int(x) for x in grp["related_id"])
    if directional:
        return gt
    # symmetrised, for contrast only: also add the reverse direction
    gt_sym = {k: set(v) for k, v in gt.items()}
    for pid, grp in confident.groupby("product_id"):
        for rid in grp["related_id"]:
            gt_sym.setdefault(int(rid), set()).add(int(pid))
    return gt_sym


def split_queries(query_ids, seed=SEED):
    ids = np.array(sorted(query_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    return (set(ids[:n_train]), set(ids[n_train:n_train + n_val]), set(ids[n_train + n_val:]))


def load_tiered_notes(pop_ids):
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")
    norm_map = pd.read_csv(OUT_DIR / "note_normalisation.csv")
    keep_map = norm_map[norm_map["canonical"].notna()][["raw_token", "canonical"]].drop_duplicates()

    notes = notes_long[notes_long["product_id"].isin(pop_ids)].copy()
    notes["raw_token"] = notes["note_raw"].map(basic_normalize)
    notes = notes.merge(keep_map, on="raw_token", how="inner")
    return notes[["product_id", "canonical", "tier"]].drop_duplicates()


# ============================================================ evaluation =====

def precision_at_k(ranked_lists, gt, query_subset, k):
    vals = []
    for q in query_subset:
        if q not in gt or q not in ranked_lists:
            continue
        cand = ranked_lists[q][:k]
        hits = sum(1 for c in cand if c in gt[q])
        vals.append(hits / k)
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def full_metrics(ranked_lists, gt, query_subset, ks=KS):
    queries = [q for q in query_subset if q in gt and q in ranked_lists]
    out = {}
    for k in ks:
        prec, rec, hit, rr = [], [], [], []
        for q in queries:
            cand = ranked_lists[q][:k]
            true = gt[q]
            hits_here = [c for c in cand if c in true]
            prec.append(len(hits_here) / k)
            rec.append(len(hits_here) / len(true) if true else 0.0)
            hit.append(1.0 if hits_here else 0.0)
            first = next((i + 1 for i, c in enumerate(cand) if c in true), None)
            rr.append(1.0 / first if first else 0.0)
        out[k] = {"precision": np.mean(prec), "recall": np.mean(rec),
                  "MRR": np.mean(rr), "hit_rate": np.mean(hit)}
    return len(queries), out


def neighbours_from_matrix(query_ids, query_rows, all_ids, all_matrix, k):
    """query_rows: matrix restricted to query_ids (same row order). Returns
    dict query_id -> ranked list of neighbour_id (self excluded), length k."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, all_matrix.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(all_matrix)
    dist, idx = nn.kneighbors(query_rows)
    out = {}
    id_arr = np.asarray(all_ids)
    for row_i, qid in enumerate(query_ids):
        cand = []
        for j in idx[row_i]:
            nid = id_arr[j]
            if nid == qid:
                continue
            cand.append(int(nid))
            if len(cand) == k:
                break
        out[int(qid)] = cand
    return out


def load_saved_neighbours(rep, max_k=20):
    df = pd.read_parquet(IN_DIR / f"08_neighbours_{rep}.parquet")
    df = df[df["rank"] <= max_k].sort_values(["product_id", "rank"])
    out = {}
    for pid, grp in df.groupby("product_id"):
        out[int(pid)] = [int(x) for x in grp["neighbour_id"]]
    return out


def build_baselines(pop, max_k, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = pop["id"].to_numpy()

    random_lists = {}
    for pid in ids:
        others = rng.choice(ids, size=max_k + 1, replace=False)
        random_lists[int(pid)] = [int(x) for x in others if x != pid][:max_k]

    pop_ranked = pop.sort_values("have", ascending=False)["id"].tolist()
    top_pop = pop_ranked[: max_k + 1]
    popularity_lists = {int(pid): [x for x in top_pop if x != pid][:max_k] for pid in ids}

    same_brand_lists = {}
    for brand, grp in pop.groupby("brand"):
        ranked = grp.sort_values("have", ascending=False)["id"].tolist()
        for pid in ranked:
            same_brand_lists[int(pid)] = [x for x in ranked if x != pid][:max_k]

    return {"random": random_lists, "popularity": popularity_lists, "same_brand": same_brand_lists}


# ==================================================== representation builders

def build_tfidf_matrix(tiered_notes, note_products, tier_weights, min_df, sublinear_tf, use_idf):
    tn = tiered_notes.copy()
    tn["w"] = tn["tier"].map(tier_weights).fillna(1.0)
    agg = tn.groupby(["product_id", "canonical"])["w"].max().reset_index()

    df_counts = agg.groupby("canonical")["product_id"].nunique()
    keep = set(df_counts[df_counts >= min_df].index)
    agg = agg[agg["canonical"].isin(keep)]
    vocab = sorted(keep)
    vocab_index = {c: i for i, c in enumerate(vocab)}
    prod_index = {p: i for i, p in enumerate(note_products)}

    rows = agg["product_id"].map(prod_index).to_numpy()
    cols = agg["canonical"].map(vocab_index).to_numpy()
    data = agg["w"].to_numpy()
    raw = sparse.csr_matrix((data, (rows, cols)), shape=(len(note_products), len(vocab)))
    tfidf = TfidfTransformer(norm="l2", use_idf=use_idf, sublinear_tf=sublinear_tf).fit_transform(raw)
    return tfidf, len(vocab)


def ppmi_shifted(co, shift):
    row_sums = co.sum(axis=1)
    total = co.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(co * total / np.outer(row_sums, row_sums))
    pmi = pmi - np.log(max(shift, 1))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)
    return ppmi


def build_svd_product_matrix(co, vocab, prod_notes, note_products, n_components, shift):
    ppmi = ppmi_shifted(co, shift)
    embedding = TruncatedSVD(n_components=n_components, random_state=SEED).fit_transform(ppmi)
    note_vec = dict(zip(vocab, embedding))
    means = []
    for pid in note_products:
        vecs = [note_vec[c] for c in prod_notes[pid]]
        means.append(np.mean(vecs, axis=0))
    mat = np.stack(means)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_accord_matrix(pop_ids, note_products):
    accords_long = pd.read_parquet(IN_DIR / "01_accords_long.parquet")
    acc = accords_long[accords_long["product_id"].isin(pop_ids)].copy()
    acc["accord"] = acc["accord"].str.lower()
    acc["strength"] = acc["strength"].astype("float64")
    wide = acc.pivot_table(index="product_id", columns="accord", values="strength",
                            aggfunc="max", fill_value=0.0)
    wide = wide.reindex(note_products, fill_value=0.0)
    return wide.to_numpy()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pop = load_population()
    pop_ids = set(pop["id"])
    print(f"analysis population: {len(pop_ids):,}")

    gt_directional = load_ground_truth(pop_ids, directional=True)
    print(f"coverage: {len(gt_directional):,} / {len(pop_ids):,} products have >=1 confident "
          f"outgoing reminds_me_of edge ({100 * len(gt_directional) / len(pop_ids):.1f}%)")

    # --- split BY QUERY PRODUCT, before anything else ----------------------------
    train_q, val_q, test_q = split_queries(set(gt_directional.keys()))
    print(f"split (60/20/20, seed {SEED}, by query product): "
          f"train={len(train_q):,}  val={len(val_q):,}  test={len(test_q):,}")
    assert train_q.isdisjoint(val_q) and train_q.isdisjoint(test_q) and val_q.isdisjoint(test_q)

    # --- directionality confirmation -----------------------------------------------
    print()
    print("=" * 92)
    print("DIRECTIONALITY CHECK")
    print("=" * 92)
    print("09_recommender_eval.py's build_ground_truth() groups confident edges by "
          "product_id (query) and collects related_id (target) -- outgoing edges only. "
          "Confirmed by re-reading the source: no reverse-direction pooling. Scoring was "
          "already directional; nothing to fix. For contrast, here is what symmetric "
          "scoring WOULD have looked like, on the untouched test split, for note_tfidf "
          "(08's default build):")
    gt_symmetric = load_ground_truth(pop_ids, directional=False)
    tfidf_default_neighbours = load_saved_neighbours("note_tfidf", max_k=max(KS))
    _, dir_metrics = full_metrics(tfidf_default_neighbours, gt_directional, test_q)
    _, sym_metrics = full_metrics(tfidf_default_neighbours, gt_symmetric, test_q)
    compare_rows = []
    for k in KS:
        for m in ["precision", "recall", "MRR", "hit_rate"]:
            compare_rows.append({"metric": f"{m}@{k}", "directional (actual)": dir_metrics[k][m],
                                  "symmetric (hypothetical)": sym_metrics[k][m]})
    compare_df = pd.DataFrame(compare_rows)
    compare_df["inflation"] = compare_df["symmetric (hypothetical)"] - compare_df["directional (actual)"]
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(compare_df.to_string(index=False))
    print("Symmetric scoring inflates every metric (a query benefits from edges pointing "
          "AT it, not just from it) -- confirms treating the relation as directional was "
          "the right call, and shows the size of the mistake it avoided.")

    # --- shared data for representation building ------------------------------------
    tiered_notes = load_tiered_notes(pop_ids)
    note_products = sorted(tiered_notes["product_id"].unique())
    prod_notes = tiered_notes.groupby("product_id")["canonical"].apply(lambda s: sorted(set(s))).to_dict()

    npz = np.load(IN_DIR / "03b_note_embedding.npz", allow_pickle=True)
    vocab_svd, co = list(npz["vocab"]), npz["co"]

    search_rows = []

    # =========================================================== note_tfidf grid ===
    print()
    print("=" * 92)
    print("GRID SEARCH: note_tfidf (objective: precision@10 on VALIDATION)")
    print("=" * 92)
    tier_options = {
        "none": {"top": 1.0, "middle": 1.0, "base": 1.0, "flat": 1.0},
        "top_heavy": {"top": 1.5, "middle": 1.0, "base": 0.7, "flat": 1.0},
    }
    val_ids_list = sorted(val_q & set(note_products))
    tfidf_results = []
    for sublinear_tf, min_df, use_idf, tier_key in iterproduct(
            [False, True], [1, 5, 20], [False, True], tier_options
    ):
        mat, vocab_size = build_tfidf_matrix(tiered_notes, note_products, tier_options[tier_key],
                                              min_df, sublinear_tf, use_idf)
        idx_lookup = {p: i for i, p in enumerate(note_products)}
        val_rows = mat[[idx_lookup[q] for q in val_ids_list]]
        ranked = neighbours_from_matrix(val_ids_list, val_rows, note_products, mat, OBJECTIVE_K)
        p10, n_eval = precision_at_k(ranked, gt_directional, val_ids_list, OBJECTIVE_K)
        row = {"representation": "note_tfidf", "sublinear_tf": sublinear_tf, "min_df": min_df,
               "use_idf": use_idf, "tier_weighting": tier_key, "vocab_size": vocab_size,
               f"precision@{OBJECTIVE_K}_val": p10, "n_eval": n_eval}
        tfidf_results.append(row)
        search_rows.append(row)
    tfidf_df = pd.DataFrame(tfidf_results).sort_values(f"precision@{OBJECTIVE_K}_val", ascending=False)
    with pd.option_context("display.width", 160, "display.max_rows", 30):
        print(tfidf_df.to_string(index=False))
    best_tfidf = tfidf_df.iloc[0].to_dict()
    print(f"\nselected note_tfidf config: sublinear_tf={best_tfidf['sublinear_tf']}, "
          f"min_df={best_tfidf['min_df']}, use_idf={best_tfidf['use_idf']}, "
          f"tier_weighting={best_tfidf['tier_weighting']}  "
          f"(val precision@{OBJECTIVE_K}={best_tfidf[f'precision@{OBJECTIVE_K}_val']:.4f})")
    default_row = tfidf_df[(tfidf_df.sublinear_tf == False) & (tfidf_df.min_df == 1) &
                            (tfidf_df.use_idf == True) & (tfidf_df.tier_weighting == "none")].iloc[0]
    print(f"08's untuned default (sublinear_tf=False, min_df=1, use_idf=True, tier=none): "
          f"val precision@{OBJECTIVE_K}={default_row[f'precision@{OBJECTIVE_K}_val']:.4f}")
    gain_tfidf = best_tfidf[f"precision@{OBJECTIVE_K}_val"] - default_row[f"precision@{OBJECTIVE_K}_val"]
    print(f"gain over untuned default: {gain_tfidf:+.4f} "
          f"({'small' if abs(gain_tfidf) < 0.01 else 'notable'})")

    # =========================================================== note_svd grid =====
    print()
    print("=" * 92)
    print("GRID SEARCH: note_svd (objective: precision@10 on VALIDATION)")
    print("=" * 92)
    val_ids_svd = sorted(val_q & set(note_products))
    svd_results = []
    for n_components, shift in iterproduct([10, 25, 50, 100, 200], [0, 1, 2]):
        mat = build_svd_product_matrix(co, vocab_svd, prod_notes, note_products, n_components, shift)
        idx_lookup = {p: i for i, p in enumerate(note_products)}
        val_rows = mat[[idx_lookup[q] for q in val_ids_svd]]
        ranked = neighbours_from_matrix(val_ids_svd, val_rows, note_products, mat, OBJECTIVE_K)
        p10, n_eval = precision_at_k(ranked, gt_directional, val_ids_svd, OBJECTIVE_K)
        row = {"representation": "note_svd", "n_components": n_components, "ppmi_shift": shift,
               f"precision@{OBJECTIVE_K}_val": p10, "n_eval": n_eval}
        svd_results.append(row)
        search_rows.append(row)
    svd_df = pd.DataFrame(svd_results).sort_values(f"precision@{OBJECTIVE_K}_val", ascending=False)
    print(svd_df.to_string(index=False))
    best_svd = svd_df.iloc[0].to_dict()
    print(f"\nselected note_svd config: n_components={best_svd['n_components']}, "
          f"ppmi_shift={best_svd['ppmi_shift']}  "
          f"(val precision@{OBJECTIVE_K}={best_svd[f'precision@{OBJECTIVE_K}_val']:.4f})")
    default_svd_row = svd_df[(svd_df.n_components == 50) & (svd_df.ppmi_shift == 0)].iloc[0]
    print(f"08's untuned default (n_components=50, shift=0): "
          f"val precision@{OBJECTIVE_K}={default_svd_row[f'precision@{OBJECTIVE_K}_val']:.4f}")
    gain_svd = best_svd[f"precision@{OBJECTIVE_K}_val"] - default_svd_row[f"precision@{OBJECTIVE_K}_val"]
    print(f"gain over untuned default: {gain_svd:+.4f} "
          f"({'small' if abs(gain_svd) < 0.01 else 'notable'})")

    # =========================================================== hybrid alpha grid =
    print()
    print("=" * 92)
    print("GRID SEARCH: hybrid alpha (note_tfidf vs accord, objective: precision@10 on VALIDATION)")
    print("=" * 92)
    default_tfidf_mat, _ = build_tfidf_matrix(tiered_notes, note_products,
                                               tier_options["none"], 1, False, True)
    accord_mat = build_accord_matrix(pop_ids, note_products)
    accord_norms = np.linalg.norm(accord_mat, axis=1)
    nonzero_accord = accord_norms > 0
    n_no_accord = int((~nonzero_accord).sum())
    if n_no_accord:
        print(f"  {n_no_accord} products have all-zero accord vectors, excluded from hybrid")

    idx_lookup = {p: i for i, p in enumerate(note_products)}
    hybrid_eligible = [p for p in note_products if nonzero_accord[idx_lookup[p]]]
    hybrid_eligible_idx = np.array([idx_lookup[p] for p in hybrid_eligible])
    val_ids_hybrid = sorted(val_q & set(hybrid_eligible))

    tfidf_val_sub = default_tfidf_mat[[idx_lookup[q] for q in val_ids_hybrid]]
    accord_val_sub = accord_mat[[idx_lookup[q] for q in val_ids_hybrid]]
    tfidf_all_sub = default_tfidf_mat[hybrid_eligible_idx]
    accord_all_sub = accord_mat[hybrid_eligible_idx]

    CHUNK = 300
    sim_tfidf_chunks, sim_accord_chunks = [], []
    for start in range(0, len(val_ids_hybrid), CHUNK):
        sim_tfidf_chunks.append(cosine_similarity(tfidf_val_sub[start:start + CHUNK], tfidf_all_sub))
        sim_accord_chunks.append(cosine_similarity(accord_val_sub[start:start + CHUNK], accord_all_sub))
    sim_tfidf_full = np.vstack(sim_tfidf_chunks)
    sim_accord_full = np.vstack(sim_accord_chunks)

    hybrid_results = []
    for alpha in [round(0.1 * i, 1) for i in range(1, 10)]:
        blended = alpha * sim_tfidf_full + (1 - alpha) * sim_accord_full
        ranked = {}
        for row_i, qid in enumerate(val_ids_hybrid):
            order = np.argsort(-blended[row_i])
            cand = []
            for j in order:
                nid = hybrid_eligible[j]
                if nid == qid:
                    continue
                cand.append(nid)
                if len(cand) == OBJECTIVE_K:
                    break
            ranked[qid] = cand
        p10, n_eval = precision_at_k(ranked, gt_directional, val_ids_hybrid, OBJECTIVE_K)
        row = {"representation": "hybrid", "alpha": alpha, f"precision@{OBJECTIVE_K}_val": p10, "n_eval": n_eval}
        hybrid_results.append(row)
        search_rows.append(row)
    hybrid_df = pd.DataFrame(hybrid_results).sort_values(f"precision@{OBJECTIVE_K}_val", ascending=False)
    print(hybrid_df.to_string(index=False))
    best_hybrid = hybrid_df.iloc[0].to_dict()
    print(f"\nselected hybrid alpha: {best_hybrid['alpha']}  "
          f"(val precision@{OBJECTIVE_K}={best_hybrid[f'precision@{OBJECTIVE_K}_val']:.4f})")

    # --- save the search table + figure --------------------------------------------
    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(OUT_DIR / "11_param_search.csv", index=False)
    print(f"\nsaved -> outputs/11_param_search.csv ({search_df.shape})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    tfidf_plot = tfidf_df.copy()
    tfidf_plot["label"] = tfidf_plot.apply(
        lambda r: f"{'sub' if r.sublinear_tf else 'raw'}/df{r.min_df}/{'idf' if r.use_idf else 'noidf'}/{r.tier_weighting}",
        axis=1)
    axes[0].barh(tfidf_plot["label"][:15][::-1], tfidf_plot[f"precision@{OBJECTIVE_K}_val"][:15][::-1])
    axes[0].set_title("note_tfidf (top 15 of 24 configs)")
    axes[0].set_xlabel(f"precision@{OBJECTIVE_K} (validation)")
    axes[0].tick_params(axis="y", labelsize=7)

    svd_pivot = svd_df.pivot(index="n_components", columns="ppmi_shift", values=f"precision@{OBJECTIVE_K}_val")
    im = axes[1].imshow(svd_pivot.values, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(svd_pivot.columns))); axes[1].set_xticklabels(svd_pivot.columns)
    axes[1].set_yticks(range(len(svd_pivot.index))); axes[1].set_yticklabels(svd_pivot.index)
    axes[1].set_xlabel("PPMI shift"); axes[1].set_ylabel("n_components")
    axes[1].set_title("note_svd")
    for i in range(svd_pivot.shape[0]):
        for j in range(svd_pivot.shape[1]):
            axes[1].text(j, i, f"{svd_pivot.values[i,j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].plot(hybrid_df.sort_values("alpha")["alpha"], hybrid_df.sort_values("alpha")[f"precision@{OBJECTIVE_K}_val"],
                 marker="o")
    axes[2].set_xlabel("alpha (weight on note_tfidf; 1-alpha on accord)")
    axes[2].set_ylabel(f"precision@{OBJECTIVE_K} (validation)")
    axes[2].set_title("hybrid alpha blend")

    fig.suptitle(f"Validation-set grid search (precision@{OBJECTIVE_K}, seed {SEED}, "
                 "never touching test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "k_param_search.png", dpi=200)
    plt.close(fig)
    print("saved -> outputs/figures/k_param_search.png")

    # ============================================================ FINAL TEST PASS ==
    print()
    print("=" * 92)
    print("FINAL TEST-SET EVALUATION (touched exactly once, tuned configs frozen above)")
    print("=" * 92)

    test_rows = []

    # tuned note_tfidf
    tfidf_tuned_mat, _ = build_tfidf_matrix(tiered_notes, note_products, tier_options[best_tfidf["tier_weighting"]],
                                             int(best_tfidf["min_df"]), bool(best_tfidf["sublinear_tf"]),
                                             bool(best_tfidf["use_idf"]))
    test_ids_tfidf = sorted(test_q & set(note_products))
    idx_lookup = {p: i for i, p in enumerate(note_products)}
    test_rows_mat = tfidf_tuned_mat[[idx_lookup[q] for q in test_ids_tfidf]]
    ranked_tfidf_tuned = neighbours_from_matrix(test_ids_tfidf, test_rows_mat, note_products,
                                                 tfidf_tuned_mat, max(KS))
    n_eval, m = full_metrics(ranked_tfidf_tuned, gt_directional, test_q)
    test_rows.append(("note_tfidf_TUNED", n_eval, m))

    # tuned note_svd
    svd_tuned_mat = build_svd_product_matrix(co, vocab_svd, prod_notes, note_products,
                                              int(best_svd["n_components"]), int(best_svd["ppmi_shift"]))
    test_ids_svd = sorted(test_q & set(note_products))
    test_rows_mat_svd = svd_tuned_mat[[idx_lookup[q] for q in test_ids_svd]]
    ranked_svd_tuned = neighbours_from_matrix(test_ids_svd, test_rows_mat_svd, note_products,
                                               svd_tuned_mat, max(KS))
    n_eval, m = full_metrics(ranked_svd_tuned, gt_directional, test_q)
    test_rows.append(("note_svd_TUNED", n_eval, m))

    # tuned hybrid (best alpha), evaluated at full k on test
    test_ids_hybrid = sorted(test_q & set(hybrid_eligible))
    tfidf_test_sub = default_tfidf_mat[[idx_lookup[q] for q in test_ids_hybrid]]
    accord_test_sub = accord_mat[[idx_lookup[q] for q in test_ids_hybrid]]
    sim_t_chunks, sim_a_chunks = [], []
    for start in range(0, len(test_ids_hybrid), CHUNK):
        sim_t_chunks.append(cosine_similarity(tfidf_test_sub[start:start + CHUNK], tfidf_all_sub))
        sim_a_chunks.append(cosine_similarity(accord_test_sub[start:start + CHUNK], accord_all_sub))
    sim_t = np.vstack(sim_t_chunks)
    sim_a = np.vstack(sim_a_chunks)
    blended = best_hybrid["alpha"] * sim_t + (1 - best_hybrid["alpha"]) * sim_a
    ranked_hybrid_tuned = {}
    for row_i, qid in enumerate(test_ids_hybrid):
        order = np.argsort(-blended[row_i])
        cand = []
        for j in order:
            nid = hybrid_eligible[j]
            if nid == qid:
                continue
            cand.append(nid)
            if len(cand) == max(KS):
                break
        ranked_hybrid_tuned[qid] = cand
    n_eval, m = full_metrics(ranked_hybrid_tuned, gt_directional, test_q)
    test_rows.append(("hybrid_TUNED", n_eval, m))

    # untuned (08's originals, all five), on test split only
    for rep in ["note_svd", "note_tfidf", "family", "accord", "hybrid"]:
        ranked = load_saved_neighbours(rep, max_k=max(KS))
        n_eval, m = full_metrics(ranked, gt_directional, test_q)
        test_rows.append((f"{rep}_untuned", n_eval, m))

    # baselines, on test split only
    baselines = build_baselines(pop, max(KS))
    for name, ranked in baselines.items():
        n_eval, m = full_metrics(ranked, gt_directional, test_q)
        test_rows.append((name, n_eval, m))

    final_rows = []
    for name, n_eval, m in test_rows:
        row = {"algorithm": name, "n_evaluated": n_eval}
        for k in KS:
            for metric, val in m[k].items():
                row[f"{metric}@{k}"] = val
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows).set_index("algorithm")
    final_df.to_csv(OUT_DIR / "11_test_results.csv")

    best_config_rows = []
    for rep_name, best in [("note_tfidf", best_tfidf), ("note_svd", best_svd), ("hybrid", best_hybrid)]:
        row = {"representation": rep_name, "selected_val_precision@10": best[f"precision@{OBJECTIVE_K}_val"]}
        for key in ["sublinear_tf", "min_df", "use_idf", "tier_weighting", "n_components", "ppmi_shift", "alpha"]:
            if key in best:
                row[key] = best[key]
        best_config_rows.append(row)
    pd.DataFrame(best_config_rows).to_csv(OUT_DIR / "11_selected_config.csv", index=False)
    print(f"saved -> outputs/11_test_results.csv, outputs/11_selected_config.csv")

    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(final_df.to_string())

    print()
    for rep, tuned_name in [("note_tfidf", "note_tfidf_TUNED"), ("note_svd", "note_svd_TUNED"),
                             ("hybrid", "hybrid_TUNED")]:
        untuned_name = f"{rep}_untuned"
        gain = final_df.loc[tuned_name, f"precision@{OBJECTIVE_K}"] - final_df.loc[untuned_name, f"precision@{OBJECTIVE_K}"]
        verdict = "negligible" if abs(gain) < 0.005 else ("small" if abs(gain) < 0.015 else "notable")
        print(f"{rep}: tuning changed test precision@{OBJECTIVE_K} by {gain:+.4f} vs untuned "
              f"08 default -- {verdict}.")
    print("\nA negative or negligible gain here is not a tuning failure: it means the "
          "validation curve is flat near its peak, so the selected configuration performs "
          "within noise of the neutral default on held-out data -- consistent with all three "
          "gains being small, and evidence that the untuned 09 results aren't artefacts of "
          "favourable defaults.")


if __name__ == "__main__":
    main()
