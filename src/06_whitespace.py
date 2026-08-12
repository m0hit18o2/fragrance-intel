"""
06_whitespace.py — white-space score per family (CLAUDE.md sections 4 & 7).

Supply = family SHARE of launches per year (never raw counts -- absolute
counts rise ~9x 2000->2025 because Fragrantica indexes recent releases more
completely; that's coverage bias, not market growth). The supply_share used
in the composite score is the family's overall share across the full
1990-2025 population; momentum is computed separately from two fixed 5-year
windows (2020-2024 vs 2015-2019), not a trailing window against the ragged
2025/2026 edge.

Demand components (approval, desire) use the BRAND-DEMEANED signals from
04_demand.py, not raw: raw approval/desire are confounded by brand-prestige
halo (e.g. a heritage house's products score high approval for being that
house's, not for their family), and a white-space read should isolate
family-level enthusiast preference net of that. Both raw and demeaned are
still carried as visible columns for transparency -- this is a modeling
choice, not one of CLAUDE.md's fixed decisions, and is flagged here for
review.

score = z(approval) + z(desire) + z(sentiment_net) - z(supply_share)
        + 0.5*z(momentum)
Weights are fixed; every component is written alongside the total.

Language discipline (CLAUDE.md section 8): "associated with", never "drives"
or "causes". Ratings are enthusiast approval, not sales. This ranking is a
set of hypotheses for concept screening, not launch decisions.

Cluster naming: clusters are named from their top 8 canonical notes by
product count (not the top-lift accord from 03_taxonomy.py -- "savory",
"terpenic" etc. are unhelpful labels). Written to outputs/cluster_names.csv.

Backbone exclusion: cluster 4 (musk, bergamot, sandalwood, vanilla, jasmine,
amber, patchouli, cedar -- ~59% of all launches) is table-stakes, not a
positioning territory, and is excluded from the ranking and from the
z-score reference population. It's reported as a separate row with its raw
component values and no score.

Outputs:
    outputs/06_territory_scores.csv
    outputs/cluster_names.csv
    data/interim/06_family_share_by_year.parquet (checkpoint for 07_figures.py)
Run standalone: python src/06_whitespace.py
(inputs: data/interim/01_products.parquet, 03_product_family.parquet,
04_family_demand.parquet, 05_family_sentiment.parquet, outputs/taxonomy_map.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
MOMENTUM_RECENT = (2020, 2024)
MOMENTUM_BASELINE = (2015, 2019)
TOP_NOTES_PER_CLUSTER = 8
# cluster 4: musk/bergamot/sandalwood/vanilla/jasmine/amber/patchouli/cedar,
# ~59% of launches -- table stakes, not positioning. See docstring.
BACKBONE_CLUSTERS = {4}


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def build_cluster_names(taxonomy_map):
    rows = []
    for c, grp in taxonomy_map.groupby("cluster"):
        top = grp.sort_values("n_products", ascending=False).head(TOP_NOTES_PER_CLUSTER)
        name = " / ".join(top["canonical_note"])
        rows.append({"cluster": c, "cluster_name": name, "top_notes": ", ".join(top["canonical_note"])})
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def main():
    products = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population", "year"])
    family = pd.read_parquet(IN_DIR / "03_product_family.parquet")
    demand = pd.read_parquet(IN_DIR / "04_family_demand.parquet")
    sentiment = pd.read_parquet(IN_DIR / "05_family_sentiment.parquet")
    taxonomy_map = pd.read_csv(OUT_DIR / "taxonomy_map.csv")

    cluster_names_df = build_cluster_names(taxonomy_map)
    cluster_names_df.to_csv(OUT_DIR / "cluster_names.csv", index=False)
    cluster_name = cluster_names_df.set_index("cluster")["cluster_name"]
    print("cluster names (top 8 notes by product count):")
    for c, name in cluster_name.items():
        print(f"  {c:>2}: {name}")
    print()

    family = family.drop(columns=["cluster_name"]).merge(cluster_names_df[["cluster", "cluster_name"]], on="cluster")

    pop = products[products["in_population"]].copy()
    fam = family.merge(pop[["id", "year"]], left_on="product_id", right_on="id", how="inner")
    print(f"in_population products: {len(pop):,}; with surviving family membership: "
          f"{fam['product_id'].nunique():,}")

    # --- supply: family share of launches per year, and overall ----------------
    by_year = fam.groupby(["year", "cluster"])["share"].sum().reset_index()
    year_totals = fam.groupby("year")["share"].sum().rename("year_total")
    by_year = by_year.merge(year_totals, on="year")
    by_year["supply_share"] = by_year["share"] / by_year["year_total"]

    overall_share = fam.groupby("cluster")["share"].sum()
    overall_supply_share = overall_share / overall_share.sum()

    print()
    print("NOTE: absolute launch counts rise sharply 2000->2025 because Fragrantica "
          "indexes recent releases far more completely -- this is coverage bias, not "
          "market growth. Supply below is always a SHARE, never a raw count.")

    def window_mean_share(lo, hi):
        sub = by_year[(by_year["year"] >= lo) & (by_year["year"] <= hi)]
        return sub.groupby("cluster")["supply_share"].mean()

    by_year_out = by_year.merge(
        family[["cluster", "cluster_name"]].drop_duplicates(), on="cluster", how="left"
    )[["year", "cluster", "cluster_name", "supply_share"]]
    by_year_out.to_parquet(IN_DIR / "06_family_share_by_year.parquet", index=False)

    recent = window_mean_share(*MOMENTUM_RECENT)
    baseline = window_mean_share(*MOMENTUM_BASELINE)
    momentum = (recent - baseline).rename("momentum")
    print(f"momentum window: mean share {MOMENTUM_RECENT[0]}-{MOMENTUM_RECENT[1]} minus "
          f"mean share {MOMENTUM_BASELINE[0]}-{MOMENTUM_BASELINE[1]}")

    # --- assemble per-family table ----------------------------------------------
    cluster_name = family[["cluster", "cluster_name"]].drop_duplicates().set_index("cluster")["cluster_name"]
    n_products = fam.groupby("cluster")["share"].sum().rename("n_products_weighted")

    df = pd.DataFrame({"cluster": sorted(cluster_name.index)}).set_index("cluster")
    df["cluster_name"] = cluster_name
    df["n_products_weighted"] = n_products
    df["supply_share"] = overall_supply_share
    df["momentum"] = momentum
    df = df.merge(demand.set_index("cluster")[
        ["approval_raw", "approval_demeaned", "desire_raw", "desire_demeaned"]
    ], left_index=True, right_index=True)
    df = df.merge(sentiment.set_index("cluster")[["net_score"]].rename(columns={"net_score": "sentiment_net"}),
                  left_index=True, right_index=True)

    df = df.reset_index()
    df["backbone"] = df["cluster"].isin(BACKBONE_CLUSTERS)
    ranked_mask = ~df["backbone"]
    n_backbone = int(df["backbone"].sum())
    print(f"backbone clusters excluded from ranking: {sorted(BACKBONE_CLUSTERS)} "
          f"({n_backbone} of {len(df)} families)")

    # --- z-score components over the non-backbone families only, composite score
    z_cols = ["z_approval", "z_desire", "z_sentiment", "z_supply", "z_momentum"]
    for col in z_cols:
        df[col] = np.nan
    df.loc[ranked_mask, "z_approval"] = zscore(df.loc[ranked_mask, "approval_demeaned"])
    df.loc[ranked_mask, "z_desire"] = zscore(df.loc[ranked_mask, "desire_demeaned"])
    df.loc[ranked_mask, "z_sentiment"] = zscore(df.loc[ranked_mask, "sentiment_net"])
    df.loc[ranked_mask, "z_supply"] = zscore(df.loc[ranked_mask, "supply_share"])
    df.loc[ranked_mask, "z_momentum"] = zscore(df.loc[ranked_mask, "momentum"])

    df["score"] = np.nan
    df.loc[ranked_mask, "score"] = (
        df.loc[ranked_mask, "z_approval"] + df.loc[ranked_mask, "z_desire"]
        + df.loc[ranked_mask, "z_sentiment"] - df.loc[ranked_mask, "z_supply"]
        + 0.5 * df.loc[ranked_mask, "z_momentum"]
    )

    ranked = df[ranked_mask].sort_values("score", ascending=False).reset_index(drop=True)
    backbone = df[~ranked_mask].reset_index(drop=True)
    df_out = pd.concat([ranked, backbone], ignore_index=True)  # ranked block, backbone row(s) last

    out_cols = [
        "cluster", "cluster_name", "backbone", "n_products_weighted",
        "approval_raw", "approval_demeaned", "desire_raw", "desire_demeaned",
        "sentiment_net", "supply_share", "momentum",
        "z_approval", "z_desire", "z_sentiment", "z_supply", "z_momentum",
        "score",
    ]
    df_out[out_cols].to_csv(OUT_DIR / "06_territory_scores.csv", index=False)

    # --- diagnostics -------------------------------------------------------------
    print()
    print("=" * 100)
    print("DIAGNOSTICS")
    print("=" * 100)
    print(f"output shape: 06_territory_scores.csv {df_out[out_cols].shape}")
    print(f"output shape: cluster_names.csv {cluster_names_df.shape}")
    print()
    print("FRAMING: these are hypotheses for concept screening, not launch decisions. "
          "Ratings reflect enthusiast approval on Fragrantica, not sales. Family signals "
          "are ASSOCIATED WITH the scores below -- not shown to drive or cause them.")
    print()
    print(f"RANKED TERRITORIES ({len(ranked)} families, backbone cluster(s) "
          f"{sorted(BACKBONE_CLUSTERS)} excluded from ranking and from the z-score "
          "reference population):")
    with pd.option_context("display.width", 240, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(ranked[out_cols].to_string(index=False))
        print()
        print("BACKBONE FAMILY (table stakes, not positioning -- raw components only, no score):")
        print(backbone[out_cols].to_string(index=False))


if __name__ == "__main__":
    main()
