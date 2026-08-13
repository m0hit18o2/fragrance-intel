"""
09_recommender_eval.py — evaluate 08's five representations against
community ground truth, plus three mandatory baselines.

Ground truth: confident (net_votes >= 5) reminds_me_of edges from 01c,
restricted to both ids in the analysis population (same filter 01c
reports on). A query product is evaluable only if it has >= 1 confident
outgoing edge -- reported explicitly as "coverage" below.

Metrics per representation/baseline, for k in {5, 10, 20}:
    precision@k, recall@k, MRR@k, hit-rate@k (>=1 true neighbour in top k)

Baselines (mandatory, not tuned):
    random       -- random other population products, seed 42
    popularity   -- global top-k by relation.have, same ranked list for
                    every query (with the query itself removed)
    same-brand   -- other products from the same brand, ranked by
                    relation.have

Nothing here is tuned to the metric -- representations are exactly what
08_recommender.py built. If a baseline beats a learned representation,
that is reported as-is.

Run standalone: python src/09_recommender_eval.py
(inputs: data/interim/01_products.parquet, 01c_similar_pairs.parquet,
08_neighbours_{rep}.parquet)
"""
from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path("data/interim")
SEED = 42
CONFIDENT_THRESHOLD = 5
KS = (5, 10, 20)
REPS = ["note_svd", "note_tfidf", "family", "accord", "hybrid"]


def build_ground_truth(pop_ids):
    sim = pd.read_parquet(IN_DIR / "01c_similar_pairs.parquet")
    rmo = sim[sim["kind"] == "reminds_me_of"].copy()
    rmo = rmo[rmo["product_id"].isin(pop_ids) & rmo["related_id"].isin(pop_ids)]
    rmo["net_votes"] = rmo["up_votes"] - rmo["down_votes"]
    confident = rmo[rmo["net_votes"] >= CONFIDENT_THRESHOLD]

    gt = {}
    for pid, grp in confident.groupby("product_id"):
        gt[int(pid)] = set(int(x) for x in grp["related_id"])
    return gt


def load_ranked_lists_from_parquet(path, max_k):
    df = pd.read_parquet(path)
    df = df[df["rank"] <= max_k].sort_values(["product_id", "rank"])
    out = {}
    for pid, grp in df.groupby("product_id"):
        out[int(pid)] = [int(x) for x in grp["neighbour_id"]]
    return out


def evaluate(ranked_lists, gt, ks=KS):
    queries = [q for q in gt if q in ranked_lists]
    n_eval = len(queries)
    results = {}
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
        results[k] = {"precision": np.mean(prec), "recall": np.mean(rec),
                      "MRR": np.mean(rr), "hit_rate": np.mean(hit)}
    return n_eval, results


def build_baselines(pop_products, max_k):
    rng = np.random.default_rng(SEED)
    ids = pop_products["id"].to_numpy()
    id_set = set(ids)

    # random: fixed random ranking of other products per query
    random_lists = {}
    for pid in ids:
        others = rng.choice(ids, size=max_k + 1, replace=False)
        random_lists[int(pid)] = [int(x) for x in others if x != pid][:max_k]

    # popularity: single global ranking by relation.have, self removed per query
    pop_ranked = pop_products.sort_values("have", ascending=False)["id"].tolist()
    top_pop = pop_ranked[: max_k + 1]
    popularity_lists = {}
    for pid in ids:
        lst = [x for x in top_pop if x != pid][:max_k]
        popularity_lists[int(pid)] = lst

    # same-brand: other products from the same brand, ranked by relation.have
    same_brand_lists = {}
    for brand, grp in pop_products.groupby("brand"):
        ranked = grp.sort_values("have", ascending=False)["id"].tolist()
        for pid in ranked:
            same_brand_lists[int(pid)] = [x for x in ranked if x != pid][:max_k]

    return {"random": random_lists, "popularity": popularity_lists, "same_brand": same_brand_lists}


def main():
    products = pd.read_parquet(IN_DIR / "01_products.parquet",
                                columns=["id", "in_population", "brand", "have"])
    pop = products[products["in_population"]].copy()
    pop["have"] = pop["have"].astype("float64")
    pop_ids = set(pop["id"])
    print(f"analysis population: {len(pop_ids):,}")

    gt = build_ground_truth(pop_ids)
    n_evaluable = len(gt)
    print(f"coverage: {n_evaluable:,} / {len(pop_ids):,} population products have "
          f">=1 confident (net_votes >= {CONFIDENT_THRESHOLD}) outgoing reminds_me_of edge "
          f"({100 * n_evaluable / len(pop_ids):.1f}%) and are therefore evaluable")
    print()

    max_k = max(KS)
    rows = []

    for rep in REPS:
        path = IN_DIR / f"08_neighbours_{rep}.parquet"
        ranked = load_ranked_lists_from_parquet(path, max_k)
        n_eval, res = evaluate(ranked, gt)
        row = {"algorithm": rep, "n_evaluated": n_eval}
        for k in KS:
            for metric, val in res[k].items():
                row[f"{metric}@{k}"] = val
        rows.append(row)

    baselines = build_baselines(pop, max_k)
    for name, ranked in baselines.items():
        n_eval, res = evaluate(ranked, gt)
        row = {"algorithm": name, "n_evaluated": n_eval}
        for k in KS:
            for metric, val in res[k].items():
                row[f"{metric}@{k}"] = val
        rows.append(row)

    table = pd.DataFrame(rows).set_index("algorithm")

    print("=" * 140)
    print("EVALUATION vs COMMUNITY GROUND TRUTH (confident reminds_me_of edges)")
    print("=" * 140)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(table.to_string())

    print()
    best_by_metric = {}
    for k in KS:
        for metric in ["precision", "recall", "MRR", "hit_rate"]:
            col = f"{metric}@{k}"
            winner = table[col].idxmax()
            is_baseline = winner in baselines
            best_by_metric[col] = (winner, table.loc[winner, col], is_baseline)

    n_baseline_wins = sum(1 for w, v, b in best_by_metric.values() if b)
    print(f"best algorithm per metric ({n_baseline_wins}/{len(best_by_metric)} metrics won by a baseline):")
    for col, (winner, val, is_baseline) in best_by_metric.items():
        tag = " <- BASELINE" if is_baseline else ""
        print(f"  {col:<12} {winner:<14} {val:.4f}{tag}")

    print()
    print("per-representation check: does any baseline beat this representation "
          "on every metric? (not just the single overall best-per-column above)")
    metric_cols = [f"{m}@{k}" for k in KS for m in ["precision", "recall", "MRR", "hit_rate"]]
    for rep in REPS:
        for base in baselines:
            beats = table.loc[base, metric_cols] > table.loc[rep, metric_cols]
            if beats.all():
                print(f"  '{base}' baseline beats '{rep}' on ALL {len(metric_cols)} metrics -- "
                      f"'{rep}' adds nothing over this baseline")
            elif beats.mean() >= 0.5:
                print(f"  '{base}' baseline beats '{rep}' on {beats.sum()}/{len(metric_cols)} metrics")


if __name__ == "__main__":
    main()
