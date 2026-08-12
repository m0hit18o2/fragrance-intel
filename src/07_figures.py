"""
07_figures.py — three figures from the finished pipeline, matplotlib only,
one chart per figure, 200 dpi PNG.

    (a) 2-D SVD scatter of canonical notes, coloured by cluster, top notes
        (by n_products) labelled.
    (b) Family launch-share over time, top 8 families by overall supply share.
    (c) Demand vs supply scatter, top 3 territories (by whitespace score)
        annotated.

Family labels use outputs/cluster_names_final.csv (the reviewed short names)
throughout, never a bare cluster number or the long auto-generated slash-name.

Run standalone: python src/07_figures.py
(inputs: outputs/taxonomy_map.csv, data/interim/06_family_share_by_year.parquet,
outputs/06_territory_scores.csv, outputs/cluster_names_final.csv)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
IN_DIR = Path("data/interim")
DPI = 200
N_LABEL_NOTES = 40
N_TOP_FAMILIES = 8
N_ANNOTATE_TERRITORIES = 3

# consistent per-cluster colours across figures (a) and (b)/(c) implicitly via
# cluster id -> tab20 colormap index
CMAP = plt.get_cmap("tab20")


def fig_a_note_map(taxonomy_map, final_names):
    fig, ax = plt.subplots(figsize=(11, 9))
    for c, grp in taxonomy_map.groupby("cluster"):
        ax.scatter(grp["svd_x"], grp["svd_y"], s=22, color=CMAP(c % 20),
                   label=final_names[c], alpha=0.8)

    top_notes = taxonomy_map.sort_values("n_products", ascending=False).head(N_LABEL_NOTES)
    for _, row in top_notes.iterrows():
        ax.annotate(row["canonical_note"], (row["svd_x"], row["svd_y"]),
                    fontsize=7, alpha=0.85, xytext=(3, 2), textcoords="offset points")

    ax.set_xlabel("SVD component 1")
    ax.set_ylabel("SVD component 2")
    ax.set_title("Olfactive note map: 2-D SVD projection of canonical-note PPMI co-occurrence\n"
                  f"coloured by cluster (k=15); top {N_LABEL_NOTES} notes by product count labelled")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="family",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "a_note_map.png", dpi=DPI)
    plt.close(fig)


def fig_b_launch_share(by_year, territory_scores, final_names):
    top8 = territory_scores.sort_values("supply_share", ascending=False).head(N_TOP_FAMILIES)
    top8_clusters = top8["cluster"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for c in top8_clusters:
        sub = by_year[by_year["cluster"] == c].sort_values("year")
        ax.plot(sub["year"], sub["supply_share"] * 100, marker="o", markersize=3,
                linewidth=1.5, color=CMAP(c % 20), label=final_names[c])

    ax.set_xlabel("year")
    ax.set_ylabel("share of launches (%)")
    ax.set_title(f"Family launch-share over time, top {N_TOP_FAMILIES} families by overall supply share\n"
                 "(share, not raw counts -- raw counts rise ~9x 2000-2025 from coverage bias, not market growth)")
    ax.legend(loc="upper left", fontsize=8, title="family", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b_launch_share_over_time.png", dpi=DPI)
    plt.close(fig)


def fig_c_demand_vs_supply(territory_scores, final_names):
    df = territory_scores.copy()
    df["demand_z"] = df["z_approval"] + df["z_desire"] + df["z_sentiment"]
    top3 = df.sort_values("score", ascending=False).head(N_ANNOTATE_TERRITORIES)

    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = 200 * (df["n_products_weighted"] / df["n_products_weighted"].max()) + 30
    sc = ax.scatter(df["supply_share"] * 100, df["demand_z"], s=sizes,
                     c=[CMAP(c % 20) for c in df["cluster"]], alpha=0.85, edgecolor="white", linewidth=0.5)

    for _, row in top3.iterrows():
        ax.annotate(final_names[row["cluster"]],
                    (row["supply_share"] * 100, row["demand_z"]),
                    fontsize=9, fontweight="bold", xytext=(6, 6), textcoords="offset points")

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("supply share (%, overall 1990-2025)")
    ax.set_ylabel("demand (z_approval + z_desire + z_sentiment, brand-demeaned)")
    ax.set_title("Demand vs supply by family\n"
                 "top 3 whitespace-score territories annotated -- hypotheses for concept "
                 "screening, not launch decisions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c_demand_vs_supply.png", dpi=DPI)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    taxonomy_map = pd.read_csv(OUT_DIR / "taxonomy_map.csv")
    by_year = pd.read_parquet(IN_DIR / "06_family_share_by_year.parquet")
    territory_scores = pd.read_csv(OUT_DIR / "06_territory_scores.csv")

    names_df = pd.read_csv(OUT_DIR / "cluster_names_final.csv")
    final_names = dict(zip(names_df["#"], names_df["Short name"]))

    fig_a_note_map(taxonomy_map, final_names)
    fig_b_launch_share(by_year, territory_scores, final_names)
    fig_c_demand_vs_supply(territory_scores, final_names)

    print("=" * 88)
    print("DIAGNOSTICS")
    print("=" * 88)
    for name in ["a_note_map.png", "b_launch_share_over_time.png", "c_demand_vs_supply.png"]:
        p = FIG_DIR / name
        print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
