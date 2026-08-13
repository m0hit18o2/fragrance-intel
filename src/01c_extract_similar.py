"""
01c_extract_similar.py — extract the similarity carousels ONLY, as a
standalone side-checkpoint. Does not touch or depend on 01_clean.py's
output logic (reads the same raw file independently).

Streams data/raw/perfumes.jsonl line by line (never json.load the whole
file) and pulls out:
    similar.reminds_me_of  -- "smells like" carousel, community-voted
    similar.also_liked     -- algorithmic carousel, no votes

Output: data/interim/01c_similar_pairs.parquet
    product_id, related_id, kind ('reminds_me_of' | 'also_liked'),
    up_votes, down_votes

Then filters to pairs where BOTH ids are in the analysis population
(joins to 01_products.parquet's in_population flag, read-only) and prints
a viability report. THIS SCRIPT STOPS AFTER REPORTING -- no downstream
script consumes 01c_similar_pairs.parquet yet.

Run standalone: python src/01c_extract_similar.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/perfumes.jsonl")
IN_DIR = Path("data/interim")
CONFIDENT_THRESHOLD = 5
APPROX_THRESHOLD = 1000  # SCHEMA.md: vote counts >= 1000 are approximate
# (the page abbreviates them), e.g. "1.2k" -- so exact counts near/above
# this are display roundings, not precise tallies.


def main():
    rows = []
    n_read = 0

    with RAW_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            rec = json.loads(line)
            pid = rec.get("id")
            similar = rec.get("similar") or {}

            for item in (similar.get("reminds_me_of") or []):
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "product_id": pid, "related_id": item.get("id"),
                    "kind": "reminds_me_of",
                    "up_votes": item.get("up_votes"), "down_votes": item.get("down_votes"),
                })
            for item in (similar.get("also_liked") or []):
                if not isinstance(item, dict):
                    continue
                rows.append({
                    "product_id": pid, "related_id": item.get("id"),
                    "kind": "also_liked",
                    "up_votes": None, "down_votes": None,
                })

    assert n_read == 131_930, f"expected 131,930 lines, read {n_read}"

    pairs = pd.DataFrame(rows)
    pairs["product_id"] = pairs["product_id"].astype("Int64")
    pairs["related_id"] = pairs["related_id"].astype("Int64")
    pairs["kind"] = pairs["kind"].astype("category")
    pairs["up_votes"] = pairs["up_votes"].astype("Int64")
    pairs["down_votes"] = pairs["down_votes"].astype("Int64")

    IN_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(IN_DIR / "01c_similar_pairs.parquet", index=False)

    total_pairs = len(pairs)
    by_kind_total = pairs["kind"].value_counts()

    # --- filter: both ids in the analysis population --------------------------
    products = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population"])
    pop_ids = set(products.loc[products["in_population"], "id"])

    surv = pairs[pairs["product_id"].isin(pop_ids) & pairs["related_id"].isin(pop_ids)].copy()
    by_kind_surv = surv["kind"].value_counts()

    # --- reminds_me_of: net_votes, confident, votes_approximate -----------------
    rmo = surv[surv["kind"] == "reminds_me_of"].copy()
    rmo["net_votes"] = rmo["up_votes"] - rmo["down_votes"]
    rmo["confident"] = rmo["net_votes"] >= CONFIDENT_THRESHOLD
    up_ge = rmo["up_votes"].fillna(-1) >= APPROX_THRESHOLD
    down_ge = rmo["down_votes"].fillna(-1) >= APPROX_THRESHOLD
    rmo["votes_approximate"] = (up_ge | down_ge).astype(bool)

    # --- symmetry: does A->B imply B->A, within the surviving set, per kind ----
    def symmetry_rate(df):
        pairs_set = set(zip(df["product_id"], df["related_id"]))
        reciprocated = sum(1 for a, b in pairs_set if (b, a) in pairs_set)
        return reciprocated, len(pairs_set)

    rmo_recip, rmo_n_unique = symmetry_rate(rmo)
    al = surv[surv["kind"] == "also_liked"]
    al_recip, al_n_unique = symmetry_rate(al)

    # --- products with >=1 confident reminds_me_of edge -------------------------
    n_pop_with_confident = rmo.loc[rmo["confident"], "product_id"].nunique()

    # --- diagnostics -------------------------------------------------------------
    print("=" * 92)
    print("DIAGNOSTICS")
    print("=" * 92)
    print(f"rows read from {RAW_PATH}: {n_read:,}")
    print(f"output shape: 01c_similar_pairs.parquet {pairs.shape}")
    print()
    print(f"total pairs extracted: {total_pairs:,}")
    print(by_kind_total.to_string())
    print()
    print(f"pairs surviving population filter (both ids in_population): {len(surv):,} "
          f"({100 * len(surv) / total_pairs:.1f}% of total)")
    print(by_kind_surv.to_string())
    print()
    print("reminds_me_of net_votes distribution (population-filtered):")
    print(rmo["net_votes"].describe().to_string())
    print()
    print(f"reminds_me_of rows with votes_approximate (>= {APPROX_THRESHOLD}, population-filtered): "
          f"{int(rmo['votes_approximate'].sum()):,} / {len(rmo):,}")
    print()
    print(f"population products with >=1 confident (net_votes >= {CONFIDENT_THRESHOLD}) "
          f"outgoing reminds_me_of edge: {n_pop_with_confident:,} "
          f"({100 * n_pop_with_confident / len(pop_ids):.1f}% of {len(pop_ids):,} population products)")
    print()
    print("symmetry (does A->B imply B->A?), population-filtered, distinct (A,B) pairs:")
    print(f"  reminds_me_of: {rmo_recip:,} / {rmo_n_unique:,} pairs reciprocated "
          f"({100 * rmo_recip / rmo_n_unique:.1f}%)" if rmo_n_unique else "  reminds_me_of: no pairs")
    print(f"  also_liked:    {al_recip:,} / {al_n_unique:,} pairs reciprocated "
          f"({100 * al_recip / al_n_unique:.1f}%)" if al_n_unique else "  also_liked: no pairs")


if __name__ == "__main__":
    main()
