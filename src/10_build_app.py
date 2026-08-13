"""
10_build_app.py — build a single self-contained two-tab HTML app.

TAB 1, Note Explorer: search any of the 492 canonical notes and see its
family, prevalence, top accords by lift, tier distribution, common
(raw co-occurrence) vs distinctive (PPMI lift) partners, cosine-similar
notes, reception (brand-demeaned approval/desire vs population), season
and community-gender skew, its family's territory-score row, example
products, and an interactive combination builder (2-5 notes).

TAB 2, Product Similarity: the existing recommender, on the TUNED hybrid
representation (alpha=0.4 * cosine(note_tfidf) + 0.6 * cosine(accord)) --
the best-performing representation/config on the held-out test split (see
outputs/11_test_results.csv). Explainability shows shared notes, shared
accords, and family overlap.

Both tabs' data is precomputed here; the browser does no PPMI/cosine/etc.
computation of its own, EXCEPT: (a) the combination builder's exact
occurrence count, which needs true set intersection over an arbitrary
2-5 note palette chosen at runtime -- precomputing that for every possible
combination is infeasible, so each note ships its population product-id
list and the browser intersects them; (b) the combination builder's
pairwise PPMI lookups and "suggested additions", which use a shipped
lookup table restricted to pairs covered by any note's top-50 PPMI
partners -- pairs outside that set report "below reporting threshold"
rather than being computed, per spec.

The full 492x492 PPMI matrix is never shipped -- only top-K partner lists
per note, plus the restricted pairwise lookup described above.

Output: outputs/app/index.html (self-contained: no CDN, no server, no
fetch -- payload is inlined as a <script> JSON blob).
Run standalone: python src/10_build_app.py
(inputs: data/interim/01_products.parquet, 01_notes_long.parquet,
01_accords_long.parquet, 04_product_demand.parquet, 03b_note_embedding.npz,
outputs/note_normalisation.csv, outputs/cluster_names_final.csv,
outputs/taxonomy_map.csv, outputs/06_territory_scores.csv)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")
APP_DIR = OUT_DIR / "app"

# tab 2 (product similarity)
HYBRID_ALPHA = 0.4  # tuned value, outputs/11_selected_config.csv
TOP_N_PRODUCTS = 3000
TOP_K_NEIGHBOURS = 20
TOP_K_ACCORDS_PRODUCT = 5
HYBRID_CHUNK = 300

# tab 1 (note explorer)
MIN_ACCORD_SUPPORT = 100  # matches 03_taxonomy.py's threshold -- avoids the
# same spurious-lift problem tiny-count accords caused there (see PROGRESS.md)
NOTE_TOP_ACCORDS = 5
NOTE_TOP_FAMILY_NOTES = 8
NOTE_PARTNERS_DISPLAY = 20   # shown in the UI
NOTE_PARTNERS_LOOKUP = 50    # used only to build the pairwise PPMI lookup
NOTE_TOP_SIMILAR = 10
NOTE_TOP_EXAMPLES = 10

TIER_ORDER = {"top": 0, "middle": 1, "base": 2, "flat": 3}
PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    """Identical to 02/03/08's basic_normalize."""
    s = raw.lower().strip()
    if PAREN_RE.search(s):
        s = PAREN_RE.sub("", s)
    return WS_RE.sub(" ", s.replace("-", " ")).strip()


def hybrid_top_neighbours(query_ids, query_tfidf, query_accord, all_ids, all_tfidf, all_accord,
                           alpha, k, chunk_size=HYBRID_CHUNK):
    """query_* restricted to query_ids rows; all_* is the full candidate pool
    (same order as all_ids). Row-chunked so no dense NxN matrix is held at
    once -- same principle as 08_recommender.py's hybrid builder."""
    id_arr = np.asarray(all_ids)
    out = {}
    n = len(query_ids)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_t = cosine_similarity(query_tfidf[start:end], all_tfidf)
        sim_a = cosine_similarity(query_accord[start:end], all_accord)
        blended = alpha * sim_t + (1 - alpha) * sim_a
        for local_i, qid in enumerate(query_ids[start:end]):
            order = np.argsort(-blended[local_i])
            cand = []
            for j in order:
                nid = id_arr[j]
                if nid == qid:
                    continue
                cand.append([int(nid), round(float(blended[local_i, j]), 3)])
                if len(cand) == k:
                    break
            out[qid] = cand
    return out


def build_tab1(pop, pop_ids):
    """Note Explorer payload: notes, ppmiLookup, population baselines, territory table."""
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")
    norm_map = pd.read_csv(OUT_DIR / "note_normalisation.csv")
    keep_map = norm_map[norm_map["canonical"].notna()][["raw_token", "canonical"]].drop_duplicates()

    notes = notes_long[notes_long["product_id"].isin(pop_ids)].copy()
    notes["raw_token"] = notes["note_raw"].map(basic_normalize)
    notes = notes.merge(keep_map, on="raw_token", how="inner")
    notes_tiered = notes[["product_id", "canonical", "tier"]].drop_duplicates()

    npz = np.load(IN_DIR / "03b_note_embedding.npz", allow_pickle=True)
    vocab, embedding, co, ppmi = list(npz["vocab"]), npz["embedding"], npz["co"], npz["ppmi"]
    notes_tiered = notes_tiered[notes_tiered["canonical"].isin(vocab)]
    print(f"tab1: note vocabulary (03b cache): {len(vocab)}")

    note_pop_list = sorted(notes_tiered["product_id"].unique())
    n_note_pop = len(note_pop_list)
    prod_pos = {p: i for i, p in enumerate(note_pop_list)}
    print(f"tab1: products with >=1 canonical note: {n_note_pop:,}")

    note_to_products = notes_tiered.groupby("canonical")["product_id"].apply(
        lambda s: sorted(int(x) for x in set(s)))

    # --- sparse products x notes indicator, vocab order -------------------------
    vocab_index = {c: i for i, c in enumerate(vocab)}
    rows_ = notes_tiered["product_id"].map(prod_pos).to_numpy()
    cols_ = notes_tiered["canonical"].map(vocab_index).to_numpy()
    X_notes = sparse.csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(n_note_pop, len(vocab)))
    note_totals = np.asarray(X_notes.sum(axis=0)).ravel()

    # --- taxonomy: family assignment, family's other top notes ------------------
    taxonomy_map = pd.read_csv(OUT_DIR / "taxonomy_map.csv")
    note_cluster = dict(zip(taxonomy_map["canonical_note"], taxonomy_map["cluster"]))
    names_df = pd.read_csv(OUT_DIR / "cluster_names_final.csv")
    short_name = {int(k): v for k, v in zip(names_df["#"], names_df["Short name"])}
    family_top_notes = {}
    for cluster_id, grp in taxonomy_map.groupby("cluster"):
        family_top_notes[int(cluster_id)] = grp.sort_values("n_products", ascending=False)["canonical_note"].tolist()

    territory = pd.read_csv(OUT_DIR / "06_territory_scores.csv")
    territory_by_cluster = territory.set_index("cluster").to_dict("index")
    territory_table = {}
    for cluster_id, row in territory_by_cluster.items():
        territory_table[int(cluster_id)] = {
            "family": short_name.get(int(cluster_id), str(cluster_id)),
            "backbone": bool(row.get("backbone", False)),
            "score": (round(float(row["score"]), 3) if pd.notna(row.get("score")) else None),
            "supplyShare": (round(float(row["supply_share"]), 4) if pd.notna(row.get("supply_share")) else None),
            "momentum": (round(float(row["momentum"]), 4) if pd.notna(row.get("momentum")) else None),
            "confidence": row.get("confidence"),
        }

    # --- tier distribution --------------------------------------------------------
    tier_counts = notes_tiered.groupby(["canonical", "tier"]).size().unstack(fill_value=0)

    # --- accord lift per note, eligible accords only (population count >= 100,
    # same threshold and reasoning as 03_taxonomy.py's cluster-level validation:
    # rare accords produce spuriously huge lift on tiny counts) -------------------
    accords_long = pd.read_parquet(IN_DIR / "01_accords_long.parquet")
    acc = accords_long[accords_long["product_id"].isin(set(note_pop_list))].copy()
    acc["accord"] = acc["accord"].str.lower()
    acc_presence = acc[["product_id", "accord"]].drop_duplicates()
    accord_counts = acc_presence.groupby("accord")["product_id"].nunique()
    eligible_accords = sorted(accord_counts[accord_counts >= MIN_ACCORD_SUPPORT].index)
    accord_base_rate = (accord_counts[eligible_accords] / n_note_pop).to_numpy()
    print(f"tab1: {len(eligible_accords)}/{len(accord_counts)} accords eligible "
          f"(population count >= {MIN_ACCORD_SUPPORT})")

    acc_pos = {a: i for i, a in enumerate(eligible_accords)}
    acc_valid = acc_presence[acc_presence["accord"].isin(acc_pos) & acc_presence["product_id"].isin(prod_pos)]
    rows_a = acc_valid["product_id"].map(prod_pos).to_numpy()
    cols_a = acc_valid["accord"].map(acc_pos).to_numpy()
    Y_accords = sparse.csr_matrix((np.ones(len(rows_a)), (rows_a, cols_a)),
                                   shape=(n_note_pop, len(eligible_accords)))
    note_accord_counts = (X_notes.T @ Y_accords).toarray()
    mention_rate = note_accord_counts / note_totals[:, None]
    lift = mention_rate / accord_base_rate[None, :]

    top_accords_per_note = {}
    for i, note in enumerate(vocab):
        order = np.argsort(-lift[i])[:NOTE_TOP_ACCORDS]
        top_accords_per_note[note] = [[eligible_accords[j], round(float(lift[i, j]), 2)] for j in order]

    # --- co-occurrence (common) / PPMI (distinctive + lookup) partners -----------
    common_partners, distinctive_partners, ppmi_lookup = {}, {}, {}
    for i, note in enumerate(vocab):
        co_row = co[i]
        order_co = np.argsort(-co_row)
        picked = [j for j in order_co if j != i and co_row[j] > 0][:NOTE_PARTNERS_DISPLAY]
        common_partners[note] = [[vocab[j], int(co_row[j])] for j in picked]

        ppmi_row = ppmi[i]
        order_ppmi = np.argsort(-ppmi_row)
        picked_display = [j for j in order_ppmi if j != i and ppmi_row[j] > 0][:NOTE_PARTNERS_DISPLAY]
        distinctive_partners[note] = [[vocab[j], round(float(ppmi_row[j]), 3)] for j in picked_display]

        picked_lookup = [j for j in order_ppmi if j != i and ppmi_row[j] > 0][:NOTE_PARTNERS_LOOKUP]
        for j in picked_lookup:
            key = "|".join(sorted([note, vocab[j]]))
            if key not in ppmi_lookup:
                ppmi_lookup[key] = round(float(ppmi_row[j]), 3)
    print(f"tab1: ppmiLookup covers {len(ppmi_lookup):,} pairs "
          f"(union of every note's top-{NOTE_PARTNERS_LOOKUP} PPMI partners)")

    # --- cosine-similar notes in the 50-d embedding -------------------------------
    cos_sim = cosine_similarity(embedding)
    similar_notes = {}
    for i, note in enumerate(vocab):
        row = cos_sim[i].copy()
        row[i] = -np.inf
        order = np.argsort(-row)[:NOTE_TOP_SIMILAR]
        similar_notes[note] = [[vocab[j], round(float(cos_sim[i, j]), 3)] for j in order]

    # --- reception: brand-demeaned approval/desire vs population ------------------
    product_demand = pd.read_parquet(IN_DIR / "04_product_demand.parquet").set_index("id")
    pop_approval_mean = float(product_demand["approval_demeaned"].mean())
    pop_desire_mean = float(product_demand["desire_demeaned"].mean())
    approval_vec = product_demand["approval_demeaned"].reindex(note_pop_list).to_numpy()
    desire_vec = product_demand["desire_demeaned"].reindex(note_pop_list).to_numpy()
    note_approval_mean = (X_notes.T @ approval_vec) / note_totals
    note_desire_mean = (X_notes.T @ desire_vec) / note_totals

    # --- season / community-gender skew, weighted by vote counts (matches 05b) ---
    # Both vote-aggregate groups can be independently null per-product (01_clean.py's
    # has_votes fix nulls a group when IT sums to zero, not when ALL groups do -- 19
    # population products have community_gender null while season is real). Null rows
    # must be EXCLUDED from that aggregation, not zero-filled (zero-filling would
    # silently claim "no one voted this way", which is a different, false claim from
    # "no vote data exists here" -- exactly the distinction 01_clean.py's fix protects).
    pop_indexed = pop.set_index("id")
    season_cols = ["season_winter", "season_spring", "season_summer", "season_autumn"]
    gender_cols = ["cg_female", "cg_female_leaning", "cg_unisex", "cg_male_leaning", "cg_male"]

    def weighted_note_shares(cols):
        raw = pop_indexed.loc[note_pop_list, cols].to_numpy(dtype="float64")
        valid = ~np.isnan(raw).any(axis=1)
        n_invalid = int((~valid).sum())
        filled = np.nan_to_num(raw, nan=0.0)
        X_valid = X_notes.multiply(valid[:, None].astype("float64")).tocsr()  # zero out null-vote product rows
        note_sum = X_valid.T @ filled
        with np.errstate(invalid="ignore", divide="ignore"):
            note_share = note_sum / note_sum.sum(axis=1, keepdims=True)
        pop_share = filled[valid].sum(axis=0) / filled[valid].sum()
        return note_share, pop_share, n_invalid

    note_season_share, pop_season_share, n_season_null = weighted_note_shares(season_cols)
    note_gender_share, pop_gender_share, n_gender_null = weighted_note_shares(gender_cols)
    print(f"tab1: {n_season_null} products excluded from season aggregation (null vote group), "
          f"{n_gender_null} from gender aggregation")

    # --- example fragrances, top by relation.have ---------------------------------
    have_arr = pop_indexed["have"].astype("float64")
    examples = {}
    for note in vocab:
        prods = sorted(note_to_products[note], key=lambda p: -have_arr.get(p, 0.0))[:NOTE_TOP_EXAMPLES]
        ex_list = []
        for p in prods:
            row = pop_indexed.loc[p]
            ex_list.append({
                "id": int(p), "n": row["name"], "b": row["brand"],
                "y": (int(row["year"]) if pd.notna(row["year"]) else None),
                "r": (round(float(row["rating_avg"]), 2) if pd.notna(row["rating_avg"]) else None),
            })
        examples[note] = ex_list

    # --- assemble per-note payload --------------------------------------------------
    def safe_pct(v):
        v = float(v)
        return round(100 * v, 1) if np.isfinite(v) else None  # None -> JSON null, never a NaN token

    season_names = ["winter", "spring", "summer", "autumn"]
    gender_names = ["female", "femaleLeaning", "unisex", "maleLeaning", "male"]
    notes_payload = {}
    for i, note in enumerate(vocab):
        cluster_id = int(note_cluster[note])
        fam_notes = [n for n in family_top_notes.get(cluster_id, []) if n != note][:NOTE_TOP_FAMILY_NOTES]
        tier_row = tier_counts.loc[note] if note in tier_counts.index else pd.Series(dtype="float64")
        tier_total = float(tier_row.sum())
        tier_pct = {t: (round(100 * float(tier_row.get(t, 0)) / tier_total, 1) if tier_total else 0.0)
                    for t in ["top", "middle", "base", "flat"]}

        notes_payload[note] = {
            "family": short_name.get(cluster_id, str(cluster_id)),
            "familyNotes": fam_notes,
            "n": int(note_totals[i]),
            "pct": round(100 * float(note_totals[i]) / n_note_pop, 2),
            "accords": top_accords_per_note[note],
            "tier": tier_pct,
            "common": common_partners[note],
            "distinctive": distinctive_partners[note],
            "similar": similar_notes[note],
            "reception": {
                "approval": round(float(note_approval_mean[i]), 4),
                "desire": round(float(note_desire_mean[i]), 4),
            },
            "season": {s: safe_pct(v) for s, v in zip(season_names, note_season_share[i])},
            "gender": {g: safe_pct(v) for g, v in zip(gender_names, note_gender_share[i])},
            "territory": territory_table.get(cluster_id, {}),
            "examples": examples[note],
            "pids": note_to_products[note],
        }

    population = {
        "approval": round(pop_approval_mean, 4),
        "desire": round(pop_desire_mean, 4),
        "season": {s: round(100 * float(v), 1) for s, v in zip(season_names, pop_season_share)},
        "gender": {g: round(100 * float(v), 1) for g, v in zip(gender_names, pop_gender_share)},
    }

    return {"notes": notes_payload, "ppmiLookup": ppmi_lookup, "population": population,
            "territory": territory_table}


def build_tab2(products, pop, pop_ids):
    """Product Similarity payload: products lookup, top_ids, neighbours (tuned hybrid)."""
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
    print(f"tab2: products with >=1 canonical note: {len(prod_notes):,}")

    fam = pd.read_parquet(IN_DIR / "03_product_family.parquet")
    names_df = pd.read_csv(OUT_DIR / "cluster_names_final.csv")
    short_name = dict(zip(names_df["#"], names_df["Short name"]))
    dominant = fam.loc[fam.groupby("product_id")["share"].idxmax()].set_index("product_id")["cluster"]
    dominant_name = dominant.map(short_name)

    accords_long = pd.read_parquet(IN_DIR / "01_accords_long.parquet")
    acc = accords_long[accords_long["product_id"].isin(pop_ids)].copy()
    acc["accord"] = acc["accord"].str.lower()
    acc["strength"] = acc["strength"].astype("float64")
    acc = acc.sort_values(["product_id", "strength"], ascending=[True, False])
    top_accords = (acc.groupby("product_id")
                   .head(TOP_K_ACCORDS_PRODUCT)
                   .groupby("product_id")
                   .apply(lambda g: list(zip(g["accord"], g["strength"].round(0).astype(int)))))

    note_products = sorted(prod_notes.index)
    canon_vocab = sorted(prod_notes.explode().dropna().unique())
    canon_index = {c: i for i, c in enumerate(canon_vocab)}
    prod_index = {p: i for i, p in enumerate(note_products)}
    rows_, cols_ = [], []
    for p in note_products:
        for c in prod_notes[p]:
            rows_.append(prod_index[p])
            cols_.append(canon_index[c])
    binary = sparse.csr_matrix((np.ones(len(rows_)), (rows_, cols_)),
                                shape=(len(note_products), len(canon_vocab)))
    tfidf_matrix = TfidfTransformer(norm="l2").fit_transform(binary)

    acc_wide = acc.pivot_table(index="product_id", columns="accord", values="strength",
                                aggfunc="max", fill_value=0.0)
    nonzero_accord = set(acc_wide.index[acc_wide.to_numpy().sum(axis=1) > 0])

    hybrid_ids = sorted(set(note_products) & nonzero_accord)
    hybrid_idx = np.array([prod_index[p] for p in hybrid_ids])
    tfidf_pool = tfidf_matrix[hybrid_idx]
    accord_pool = acc_wide.reindex(hybrid_ids, fill_value=0.0).to_numpy()
    print(f"tab2: hybrid-eligible products (note_tfidf ∩ accord): {len(hybrid_ids):,}")

    eligible = pop[pop["id"].isin(hybrid_ids)].copy()
    eligible["have"] = eligible["have"].astype("float64")
    top_ids = eligible.sort_values("have", ascending=False).head(TOP_N_PRODUCTS)["id"].tolist()
    print(f"tab2: top {TOP_N_PRODUCTS} products by relation.have (hybrid-eligible): {len(top_ids):,}")

    hybrid_pos = {p: i for i, p in enumerate(hybrid_ids)}
    query_idx = [hybrid_pos[p] for p in top_ids]
    neighbours_payload = hybrid_top_neighbours(
        top_ids, tfidf_pool[query_idx], accord_pool[query_idx],
        hybrid_ids, tfidf_pool, accord_pool, HYBRID_ALPHA, TOP_K_NEIGHBOURS,
    )
    neighbours_payload = {int(k): v for k, v in neighbours_payload.items()}

    referenced_ids = set(top_ids)
    for pairs in neighbours_payload.values():
        referenced_ids.update(n for n, _ in pairs)
    print(f"tab2: distinct products referenced (top_ids + all their neighbours): {len(referenced_ids):,}")

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
        print(f"tab2 WARNING: {n_missing_notes} referenced products have no notes (unexpected)")

    return {
        "meta": {
            "representation": "hybrid",
            "alpha": HYBRID_ALPHA,
            "note": (f"hybrid: {HYBRID_ALPHA}*cosine(note_tfidf) + {1 - HYBRID_ALPHA:.1f}*cosine(accord) "
                     f"-- tuned alpha per 11_param_optimisation.py's validation grid search; "
                     f"best-performing representation on the held-out test split "
                     f"(see outputs/11_test_results.csv)"),
            "n_top_products": len(top_ids),
            "n_products_total": len(products_payload),
        },
        "products": products_payload,
        "top_ids": top_ids,
        "neighbours": neighbours_payload,
    }


def main():
    APP_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_parquet(
        IN_DIR / "01_products.parquet",
        columns=["id", "in_population", "name", "brand", "year", "gender", "rating_avg", "have",
                 "season_winter", "season_spring", "season_summer", "season_autumn",
                 "cg_female", "cg_female_leaning", "cg_unisex", "cg_male_leaning", "cg_male"],
    )
    pop = products[products["in_population"]].copy()
    pop_ids = set(pop["id"])
    print(f"analysis population: {len(pop_ids):,}")
    print()

    print("=" * 88)
    print("TAB 1: NOTE EXPLORER")
    print("=" * 88)
    tab1 = build_tab1(pop, pop_ids)

    print()
    print("=" * 88)
    print("TAB 2: PRODUCT SIMILARITY")
    print("=" * 88)
    tab2 = build_tab2(products, pop, pop_ids)

    payload = {
        "meta": {"n_notes": len(tab1["notes"]), "n_top_products": tab2["meta"]["n_top_products"]},
        "tab1": tab1,
        "tab2": tab2,
    }

    # allow_nan=False: fail loudly at build time if a NaN/Infinity ever leaks into the
    # payload (Python's json module writes literal NaN/Infinity tokens by default,
    # which are NOT valid JSON and silently break JSON.parse in the browser -- this
    # is exactly how the community-gender null-group bug surfaced during testing).
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    payload_json = payload_json.replace("</", "<\\/")
    payload_size_mb = len(payload_json.encode("utf-8")) / (1024 * 1024)
    print()
    print(f"total payload: {payload_size_mb:.2f} MB (JSON, minified)")

    html = build_html(payload_json)
    out_path = APP_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    total_mb = out_path.stat().st_size / (1024 * 1024)

    print()
    print("=" * 88)
    print("DIAGNOSTICS")
    print("=" * 88)
    print(f"output: {out_path}  ({total_mb:.2f} MB)")
    print(f"tab1: {len(tab1['notes']):,} notes, {len(tab1['ppmiLookup']):,} ppmiLookup pairs, "
          f"{len(tab1['territory']):,} territory rows")
    print(f"tab2: top_ids={len(tab2['top_ids']):,}, products={len(tab2['products']):,}, "
          f"neighbour lists={len(tab2['neighbours']):,}")


def build_html(payload_json):
    template_path = Path(__file__).parent / "10_app_template.html"
    template = template_path.read_text(encoding="utf-8")
    return template.replace("__PAYLOAD_JSON__", payload_json)


if __name__ == "__main__":
    main()
