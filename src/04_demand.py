"""
04_demand.py — three (four, counting value) demand signals per CLAUDE.md
section 3, computed separately and never averaged into one number here.

    Approval = Bayesian-shrunk rating: (v*R + m*C)/(v+m)
        v = people, R = rating_avg, C = mean rating_avg over in_population,
        m = median people over in_population.
    Desire   = want / (have+had+want); NULL when have+had+want == 0 or any
               of have/had/want is itself missing (no relation votes).
    Adoption = log1p(have); NULL when have is missing.
    Value    = price_value_avg (as-is).

Brand-demeaning: within brands with >=5 in_population products, subtract the
brand mean from each signal (computed on non-null values). Smaller brands
keep raw values (brand_demeaned=False). Both raw and demeaned versions are
aggregated to family level, weighted by fractional family membership.

Output: data/interim/04_family_demand.parquet
        data/interim/04_product_demand.parquet (per-product signals, for
        anything downstream that needs product-level rather than
        family-level demand -- e.g. 10_build_app.py's per-note reception stats)
Run standalone: python src/04_demand.py
(inputs: data/interim/01_products.parquet, 03_product_family.parquet)
"""
from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path("data/interim")
MIN_BRAND_PRODUCTS = 5


def weighted_mean_skipna(values, weights):
    mask = values.notna()
    w = weights[mask]
    v = values[mask]
    wsum = w.sum()
    if wsum == 0:
        return np.nan, 0.0
    return float((v * w).sum() / wsum), float(wsum)


def main():
    products = pd.read_parquet(IN_DIR / "01_products.parquet")
    family = pd.read_parquet(IN_DIR / "03_product_family.parquet")

    pop = products[products["in_population"]].copy()
    print(f"in_population products: {len(pop):,}")

    # --- four signals, product level -----------------------------------------
    people = pop["people"].astype("float64")
    rating = pop["rating_avg"].astype("float64")
    have = pop["have"].astype("float64")
    had = pop["had"].astype("float64")
    want = pop["want"].astype("float64")

    m = people.median()
    C = rating.mean()
    print(f"Bayesian shrinkage: m (median people, in_population) = {m:.1f}, "
          f"C (mean rating_avg, in_population) = {C:.4f}")

    pop["approval"] = (people * rating + m * C) / (people + m)

    denom = have + had + want
    relation_present = have.notna() & had.notna() & want.notna()
    desire = pd.Series(np.nan, index=pop.index)
    valid = relation_present & (denom > 0)
    desire[valid] = want[valid] / denom[valid]
    pop["desire"] = desire

    pop["adoption"] = np.log1p(have)  # NaN propagates automatically when have is NaN

    pop["value"] = pop["price_value_avg"].astype("float64")

    signals = ["approval", "desire", "adoption", "value"]
    for s in signals:
        n_null = pop[s].isna().sum()
        print(f"  {s:<10} null: {n_null:,} ({100 * n_null / len(pop):.1f}%)")

    # --- brand demeaning -------------------------------------------------------
    brand_counts = pop.groupby("brand")["id"].transform("nunique")
    pop["brand_demeaned"] = brand_counts >= MIN_BRAND_PRODUCTS
    n_brands_ge5 = (pop.groupby("brand")["id"].nunique() >= MIN_BRAND_PRODUCTS).sum()
    n_brands_total = pop["brand"].nunique()
    print(f"brands: {n_brands_total:,} total in population, {n_brands_ge5:,} with "
          f">={MIN_BRAND_PRODUCTS} products (demeaned); "
          f"{(pop['brand_demeaned']).mean() * 100:.1f}% of products demeaned")

    for s in signals:
        brand_mean = pop.groupby("brand")[s].transform("mean")  # skips NaN
        demeaned = pop[s] - brand_mean
        pop[f"{s}_demeaned"] = np.where(pop["brand_demeaned"], demeaned, pop[s])

    # --- per-product signals, saved so nothing downstream (e.g. 10_build_app.py's
    # per-note reception stats) needs to re-derive this formula from scratch -----
    product_demand_cols = ["id"] + signals + [f"{s}_demeaned" for s in signals] + ["brand_demeaned"]
    pop[product_demand_cols].to_parquet(IN_DIR / "04_product_demand.parquet", index=False)

    # --- aggregate to family level, weighted by fractional membership ----------
    fam = family.merge(
        pop[["id"] + signals + [f"{s}_demeaned" for s in signals]],
        left_on="product_id", right_on="id", how="inner",
    )
    cluster_name = family[["cluster", "cluster_name"]].drop_duplicates().set_index("cluster")["cluster_name"]

    rows = []
    for c, grp in fam.groupby("cluster"):
        row = {"cluster": c, "cluster_name": cluster_name.get(c), "n_products_weighted": grp["share"].sum()}
        for s in signals:
            raw_mean, raw_w = weighted_mean_skipna(grp[s], grp["share"])
            dem_mean, dem_w = weighted_mean_skipna(grp[f"{s}_demeaned"], grp["share"])
            row[f"{s}_raw"] = raw_mean
            row[f"{s}_raw_weight"] = raw_w
            row[f"{s}_demeaned"] = dem_mean
            row[f"{s}_demeaned_weight"] = dem_w
        rows.append(row)

    family_demand = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    family_demand.to_parquet(IN_DIR / "04_family_demand.parquet", index=False)

    # --- diagnostics -------------------------------------------------------------
    print()
    print("=" * 100)
    print("DIAGNOSTICS")
    print("=" * 100)
    print(f"output shape: 04_family_demand.parquet {family_demand.shape}")
    print()
    display_cols = ["cluster", "cluster_name", "n_products_weighted",
                     "approval_raw", "approval_demeaned",
                     "desire_raw", "desire_demeaned",
                     "adoption_raw", "adoption_demeaned",
                     "value_raw", "value_demeaned"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(family_demand[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
