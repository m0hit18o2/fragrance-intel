"""
05b_seasonality.py — native seasonality signal (CLAUDE.md section 6).

Per family, share of season votes (winter/spring/summer/autumn), weighted by
fractional family membership: each product's raw season vote counts are
multiplied by its family-membership share, summed across products, then
renormalized to a share within each family. Products with no season votes
(the vote-aggregate blob was absent or all-zero, see 01_clean.py's
has_votes fix) contribute nothing and are excluded.

Output: data/interim/05b_family_season.parquet
Run standalone: python src/05b_seasonality.py
(inputs: data/interim/01_products.parquet, 03_product_family.parquet)
"""
from pathlib import Path

import pandas as pd

IN_DIR = Path("data/interim")
SEASON_COLS = ["season_winter", "season_spring", "season_summer", "season_autumn"]


def main():
    products = pd.read_parquet(IN_DIR / "01_products.parquet")
    family = pd.read_parquet(IN_DIR / "03_product_family.parquet")

    pop = products[products["in_population"]].copy()
    has_seasons = pop[SEASON_COLS].notna().all(axis=1)
    n_no_seasons = int((~has_seasons).sum())
    print(f"in_population products: {len(pop):,}")
    print(f"  with season votes (has_votes on the seasons blob): {int(has_seasons.sum()):,}")
    print(f"  without season votes (excluded from this signal): {n_no_seasons:,}")

    pop = pop[has_seasons].copy()
    for c in SEASON_COLS:
        pop[c] = pop[c].astype("float64")

    fam = family.merge(pop[["id"] + SEASON_COLS], left_on="product_id", right_on="id", how="inner")
    cluster_name = family[["cluster", "cluster_name"]].drop_duplicates().set_index("cluster")["cluster_name"]

    rows = []
    for c, grp in fam.groupby("cluster"):
        weighted_votes = {}
        for season_col in SEASON_COLS:
            weighted_votes[season_col] = (grp[season_col] * grp["share"]).sum()
        total = sum(weighted_votes.values())
        row = {"cluster": c, "cluster_name": cluster_name.get(c), "total_weighted_votes": total}
        for season_col in SEASON_COLS:
            season = season_col.replace("season_", "")
            row[f"{season}_share"] = weighted_votes[season_col] / total if total > 0 else float("nan")
        rows.append(row)

    family_season = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    family_season.to_parquet(IN_DIR / "05b_family_season.parquet", index=False)

    print()
    print("=" * 100)
    print("DIAGNOSTICS")
    print("=" * 100)
    print(f"output shape: 05b_family_season.parquet {family_season.shape}")
    print()
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(family_season.to_string(index=False))


if __name__ == "__main__":
    main()
