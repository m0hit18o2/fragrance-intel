"""
10_build_app.py — build a single self-contained HTML app for browsing the
top 3,000 products (by relation.have) and their nearest neighbours.

Uses note_tfidf, the best-performing representation per
09_recommender_eval.py (wins all 12 precision/recall/MRR/hit-rate metrics),
for the top-20 neighbours.

Payload design (kept compact, under the ~8MB budget):
    products:   dict, id -> {name, brand, year, gender, rating, family,
                notes (canonical list), accords (top 3, [name, strength])}.
                Covers the UNION of the top 3,000 products and every
                distinct product that appears as one of their neighbours
                -- each product's metadata is stored exactly once no
                matter how many neighbour lists reference it.
    top_ids:    the 3,000 queryable product ids, sorted by relation.have desc.
    neighbours: dict, id -> [[neighbour_id, similarity], ...] (<=20), for
                the 3,000 top_ids only.

Explainability (shared notes, family overlap) is NOT precomputed or
embedded per neighbour -- both the query and neighbour product's full
metadata (notes, family) is already in `products`, so the app computes
the shared-notes intersection and same-family comparison client-side at
render time from data already needed for display. This avoids a second
copy of every product's notes inside every neighbour list.

Output: outputs/app/index.html (self-contained: no CDN, no server, no
fetch -- payload is inlined as a <script> JSON blob).
Run standalone: python src/10_build_app.py
(inputs: data/interim/01_products.parquet, 01_notes_long.parquet,
01_accords_long.parquet, 03_product_family.parquet,
08_neighbours_note_tfidf.parquet, outputs/note_normalisation.csv,
outputs/cluster_names_final.csv)
"""
import json
import re
from pathlib import Path

import pandas as pd

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
APP_DIR = OUT_DIR / "app"
BEST_REP = "note_tfidf"  # per 09_recommender_eval.py
TOP_N_PRODUCTS = 3000
TOP_K_NEIGHBOURS = 20
TOP_K_ACCORDS = 3

TIER_ORDER = {"top": 0, "middle": 1, "base": 2, "flat": 3}
PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    """Identical to 02/03/08's basic_normalize."""
    s = raw.lower().strip()
    if PAREN_RE.search(s):
        s = PAREN_RE.sub("", s)
    return WS_RE.sub(" ", s.replace("-", " ")).strip()


def main():
    APP_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_parquet(
        IN_DIR / "01_products.parquet",
        columns=["id", "in_population", "name", "brand", "year", "gender", "rating_avg", "have"],
    )
    pop = products[products["in_population"]].copy()
    pop_ids = set(pop["id"])

    # --- canonical notes per product, tier-ordered (first tier a note appears in wins) --
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")
    norm_map = pd.read_csv(OUT_DIR / "note_normalisation.csv")
    keep_map = norm_map[norm_map["canonical"].notna()][["raw_token", "canonical"]].drop_duplicates()

    notes = notes_long[notes_long["product_id"].isin(pop_ids)].copy()
    notes["raw_token"] = notes["note_raw"].map(basic_normalize)
    notes = notes.merge(keep_map, on="raw_token", how="inner")
    notes["tier_rank"] = notes["tier"].map(TIER_ORDER)
    notes = notes.sort_values(["product_id", "tier_rank"])
    prod_notes = (notes.drop_duplicates(["product_id", "canonical"])
                  .groupby("product_id")["canonical"].apply(list))
    print(f"products with >=1 canonical note: {len(prod_notes):,}")

    # --- dominant family per product ------------------------------------------------
    fam = pd.read_parquet(IN_DIR / "03_product_family.parquet")
    names_df = pd.read_csv(OUT_DIR / "cluster_names_final.csv")
    short_name = dict(zip(names_df["#"], names_df["Short name"]))
    dominant = fam.loc[fam.groupby("product_id")["share"].idxmax()].set_index("product_id")["cluster"]
    dominant_name = dominant.map(short_name)

    # --- top-3 accords per product, by strength --------------------------------------
    accords_long = pd.read_parquet(IN_DIR / "01_accords_long.parquet")
    acc = accords_long[accords_long["product_id"].isin(pop_ids)].copy()
    acc["accord"] = acc["accord"].str.lower()
    acc["strength"] = acc["strength"].astype("float64")
    acc = acc.sort_values(["product_id", "strength"], ascending=[True, False])
    top_accords = (acc.groupby("product_id")
                   .head(TOP_K_ACCORDS)
                   .groupby("product_id")
                   .apply(lambda g: list(zip(g["accord"], g["strength"].round(0).astype(int)))))

    # --- neighbours from the best representation --------------------------------------
    nbrs = pd.read_parquet(IN_DIR / f"08_neighbours_{BEST_REP}.parquet")
    nbrs = nbrs[nbrs["rank"] <= TOP_K_NEIGHBOURS].sort_values(["product_id", "rank"])
    note_tfidf_ids = set(nbrs["product_id"].unique())

    # --- top 3,000 by relation.have, restricted to products with neighbours -----------
    eligible = pop[pop["id"].isin(note_tfidf_ids)].copy()
    eligible["have"] = eligible["have"].astype("float64")
    top_ids = eligible.sort_values("have", ascending=False).head(TOP_N_PRODUCTS)["id"].tolist()
    print(f"top {TOP_N_PRODUCTS} products by relation.have (with neighbours available): {len(top_ids):,}")

    neighbours_payload = {}
    referenced_ids = set(top_ids)
    for pid, grp in nbrs[nbrs["product_id"].isin(top_ids)].groupby("product_id"):
        pairs = [[int(r["neighbour_id"]), round(float(r["similarity"]), 3)] for _, r in grp.iterrows()]
        neighbours_payload[int(pid)] = pairs
        referenced_ids.update(n for n, _ in pairs)

    print(f"distinct products referenced (top_ids + all their neighbours): {len(referenced_ids):,}")

    # --- build compact per-product metadata for every referenced product --------------
    meta = products.set_index("id")
    products_payload = {}
    n_missing_notes = 0
    for pid in referenced_ids:
        row = meta.loc[pid]
        note_list = prod_notes.get(pid, [])
        if not note_list:
            n_missing_notes += 1
        rating = row["rating_avg"]
        products_payload[str(pid)] = {
            "n": row["name"], "b": row["brand"],
            "y": (int(row["year"]) if pd.notna(row["year"]) else None),
            "g": row["gender"],
            "r": (round(float(rating), 2) if pd.notna(rating) else None),
            "f": dominant_name.get(pid),
            "notes": note_list,
            "acc": [[a, s] for a, s in top_accords.get(pid, [])],
        }
    if n_missing_notes:
        print(f"WARNING: {n_missing_notes} referenced products have no notes (unexpected)")

    payload = {
        "meta": {
            "representation": BEST_REP,
            "note": "best-performing representation per 09_recommender_eval.py (wins all 12 metrics)",
            "n_top_products": len(top_ids),
            "n_products_total": len(products_payload),
        },
        "products": products_payload,
        "top_ids": top_ids,
        "neighbours": neighbours_payload,
    }

    payload_json = json.dumps(payload, separators=(",", ":"))
    # defend against a product name/brand containing "</script" and breaking the
    # HTML parser's scan for the script tag's end -- "\/" is a legal JSON escape
    # for "/", so this is still valid JSON that JSON.parse reads identically.
    payload_json = payload_json.replace("</", "<\\/")
    payload_size_mb = len(payload_json.encode("utf-8")) / (1024 * 1024)
    print(f"payload: {len(products_payload):,} products, {len(neighbours_payload):,} neighbour lists, "
          f"{payload_size_mb:.2f} MB (JSON, minified)")

    html = build_html(payload_json)
    out_path = APP_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    total_mb = out_path.stat().st_size / (1024 * 1024)

    print()
    print("=" * 88)
    print("DIAGNOSTICS")
    print("=" * 88)
    print(f"output: {out_path}  ({total_mb:.2f} MB)")
    print(f"top_ids: {len(top_ids):,}  |  products in payload: {len(products_payload):,}  |  "
          f"neighbour lists: {len(neighbours_payload):,}")


def build_html(payload_json):
    template_path = Path(__file__).parent / "10_app_template.html"
    template = template_path.read_text(encoding="utf-8")
    return template.replace("__PAYLOAD_JSON__", payload_json)


if __name__ == "__main__":
    main()
