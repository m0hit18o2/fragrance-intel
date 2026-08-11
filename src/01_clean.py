"""
01_clean.py — stream data/raw/perfumes.jsonl and flatten into four parquet
checkpoints under data/interim/.

Reads the raw dump line by line (never loads the whole file into memory),
extracts one products table plus three long/tidy tables (notes, accords,
ai_summary pros/cons), and computes an `in_population` flag per CLAUDE.md
section 0.

Outputs:
    data/interim/01_products.parquet
    data/interim/01_notes_long.parquet
    data/interim/01_accords_long.parquet
    data/interim/01_ai_summary_long.parquet

Run standalone: python src/01_clean.py
"""
import json
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/perfumes.jsonl")
OUT_DIR = Path("data/interim")

# --- fixed analytical decisions, CLAUDE.md section 0 ------------------------
# people threshold: confirmed at 50 (median people across the corpus is only
# 15; the surviving-row table at 20/50/100/200 was reviewed and 50 was chosen).
PEOPLE_THRESHOLD = 50
YEAR_MIN, YEAR_MAX = 1990, 2025


def dig(rec, *keys):
    """Safe nested dict.get() chain; None if any level is missing/not a dict."""
    cur = rec
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def extract_notes(rec):
    """Return [(tier, note_raw), ...]. Tiered XOR flat, never both (per SCHEMA.md)."""
    notes = rec.get("notes") or {}
    tiered = notes.get("tiered") or {}
    flat = notes.get("flat") or []
    has_tiered = any(tiered.get(t) for t in ("top", "middle", "base"))

    out = []
    if has_tiered:
        for tier in ("top", "middle", "base"):
            for note in (tiered.get(tier) or []):
                name = note.get("name") if isinstance(note, dict) else None
                if name and name.strip():
                    out.append((tier, name.strip()))
    else:
        for note in flat:
            name = note.get("name") if isinstance(note, dict) else None
            if name and name.strip():
                out.append(("flat", name.strip()))
    return out


# --- vote-aggregate dicts: SCHEMA.md says these (+ people) go null TOGETHER
# when the source blob is absent. Empirically (checked against the raw dump)
# that is not reliable in this corpus -- e.g. `people` is null on records
# where relation/seasons/daypart/community_gender/longevity/sillage/price_value
# are still present as real (all-zero) dicts. So each group is extracted from
# its OWN presence, never inferred from another field's presence/absence.
#
# Within a present group, an all-zero dict means "nobody voted on this",
# not "the true value is zero" (e.g. relation have=had=want=0 must not be
# read as desire = 0/0 = 0). Each *_fields() helper below nulls out its
# group's values when the dict sums to zero, and reports whether it found
# real (nonzero) votes so build_product_row can set has_votes.

def relation_fields(rec):
    d = rec.get("relation")
    if not d:
        return None, None, None, False
    have, had, want = d.get("have"), d.get("had"), d.get("want")
    if (have or 0) + (had or 0) + (want or 0) == 0:
        return None, None, None, False
    return have, had, want, True


def community_gender_fields(rec):
    d = rec.get("community_gender")
    if not d:
        return None, None, None, None, None, False
    f, fl, u, ml, m = (d.get("female"), d.get("female_leaning"), d.get("unisex"),
                       d.get("male_leaning"), d.get("male"))
    if (f or 0) + (fl or 0) + (u or 0) + (ml or 0) + (m or 0) == 0:
        return None, None, None, None, None, False
    return f, fl, u, ml, m, True


def seasons_fields(rec):
    d = rec.get("seasons")
    if not d:
        return None, None, None, None, False
    w, sp, su, a = d.get("winter"), d.get("spring"), d.get("summer"), d.get("autumn")
    if (w or 0) + (sp or 0) + (su or 0) + (a or 0) == 0:
        return None, None, None, None, False
    return w, sp, su, a, True


def daypart_fields(rec):
    d = rec.get("daypart")
    if not d:
        return None, None, False
    day, night = d.get("day"), d.get("night")
    if (day or 0) + (night or 0) == 0:
        return None, None, False
    return day, night, True


def dict_avg(rec, key):
    """Extract `<key>.average`; None if that vote-aggregate dict is absent/empty."""
    d = rec.get(key)
    if not d:
        return None
    return d.get("average")


def build_product_row(rec, notes):
    have, had, want, rel_voted = relation_fields(rec)
    cg_f, cg_fl, cg_u, cg_ml, cg_m, cg_voted = community_gender_fields(rec)
    s_w, s_sp, s_su, s_a, season_voted = seasons_fields(rec)
    dp_d, dp_n, dp_voted = daypart_fields(rec)
    pros = dig(rec, "ai_summary", "pros") or []
    cons = dig(rec, "ai_summary", "cons") or []

    return {
        "id": rec.get("id"),
        "slug": rec.get("slug"),
        "name": rec.get("name"),
        "brand": rec.get("brand"),
        "year": rec.get("year"),
        "gender": rec.get("gender"),
        "collection": rec.get("collection"),
        "rating_avg": dig(rec, "rating", "average"),
        "people": rec.get("people"),
        "have": have,
        "had": had,
        "want": want,
        "price_value_avg": dict_avg(rec, "price_value"),
        "longevity_avg": dict_avg(rec, "longevity"),
        "sillage_avg": dict_avg(rec, "sillage"),
        "season_winter": s_w,
        "season_spring": s_sp,
        "season_summer": s_su,
        "season_autumn": s_a,
        "daypart_day": dp_d,
        "daypart_night": dp_n,
        "cg_female": cg_f,
        "cg_female_leaning": cg_fl,
        "cg_unisex": cg_u,
        "cg_male_leaning": cg_ml,
        "cg_male": cg_m,
        "n_notes": len(notes),
        "has_ai_summary": bool(pros) or bool(cons),
        "has_votes": rel_voted or cg_voted or season_voted or dp_voted,
        "in_population": False,  # placeholder, set vectorized after streaming
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    products, notes_rows, accords_rows, ai_rows = [], [], [], []
    n_read = 0
    n_bad_json = 0

    with RAW_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue

            pid = rec.get("id")
            notes = extract_notes(rec)
            products.append(build_product_row(rec, notes))

            for tier, note_raw in notes:
                notes_rows.append({"product_id": pid, "tier": tier, "note_raw": note_raw})

            for acc in (rec.get("accords") or []):
                if isinstance(acc, dict):
                    accords_rows.append({
                        "product_id": pid,
                        "accord": acc.get("name"),
                        "strength": acc.get("strength"),
                    })

            for item in (dig(rec, "ai_summary", "pros") or []):
                if isinstance(item, dict):
                    ai_rows.append({
                        "product_id": pid, "polarity": "pro",
                        "text": item.get("text"),
                        "up_votes": item.get("up_votes"),
                        "down_votes": item.get("down_votes"),
                    })
            for item in (dig(rec, "ai_summary", "cons") or []):
                if isinstance(item, dict):
                    ai_rows.append({
                        "product_id": pid, "polarity": "con",
                        "text": item.get("text"),
                        "up_votes": item.get("up_votes"),
                        "down_votes": item.get("down_votes"),
                    })

    assert n_read == 131_930, f"expected 131,930 lines, read {n_read}"
    assert n_bad_json == 0, f"{n_bad_json} lines failed json.loads"

    products_df = pd.DataFrame(products)

    # --- dtypes: nullable Int64 for count-like fields (real null, no NaN-float
    # display noise), float64 for averages, bool for flags.
    int_cols = [
        "id", "year", "people", "have", "had", "want",
        "season_winter", "season_spring", "season_summer", "season_autumn",
        "daypart_day", "daypart_night",
        "cg_female", "cg_female_leaning", "cg_unisex", "cg_male_leaning", "cg_male",
        "n_notes",
    ]
    for c in int_cols:
        products_df[c] = products_df[c].astype("Int64")

    float_cols = ["rating_avg", "price_value_avg", "longevity_avg", "sillage_avg"]
    for c in float_cols:
        products_df[c] = products_df[c].astype("float64")

    products_df["has_ai_summary"] = products_df["has_ai_summary"].astype(bool)
    products_df["has_votes"] = products_df["has_votes"].astype(bool)

    # --- population diagnostics: exploratory table across people thresholds -
    # Kleene-logic-safe boolean conditions (NA -> False explicitly, never
    # silently coerced by comparison operators on nullable Int64).
    has_note = (products_df["n_notes"] >= 1).astype(bool)
    year_notna = products_df["year"].notna()
    year_ok = (year_notna & (products_df["year"] >= YEAR_MIN) &
               (products_df["year"] <= YEAR_MAX)).fillna(False).astype(bool)

    base_mask = has_note & year_ok  # "real analysis population" restriction, minus people
    base = products_df[base_mask]

    def people_ok_mask(df, threshold):
        if threshold is None:
            return pd.Series(True, index=df.index)
        return (df["people"].notna() & (df["people"] >= threshold)).fillna(False).astype(bool)

    table_rows = []
    for label, thr in [("no threshold", None), (20, 20), (50, 50), (100, 100), (200, 200)]:
        subset = base[people_ok_mask(base, thr)]
        n = len(subset)
        brand_counts = subset["brand"].value_counts()
        table_rows.append({
            "people>=": label,
            "rows": n,
            "% with ai_summary": round(100 * subset["has_ai_summary"].mean(), 1) if n else 0.0,
            "% with year": round(100 * subset["year"].notna().mean(), 1) if n else 0.0,
            "distinct brands": int(subset["brand"].nunique()),
            "brands with >=5 products": int((brand_counts >= 5).sum()),
        })
    pop_table = pd.DataFrame(table_rows)

    print("=" * 88)
    print("POPULATION THRESHOLD EXPLORATION")
    print("(all rows already restricted to >=1 note AND year in "
          f"[{YEAR_MIN}, {YEAR_MAX}] -- '% with year' is a sanity check, expected ~100%)")
    print("=" * 88)
    print(pop_table.to_string(index=False))
    print()

    # --- final in_population flag, threshold = PEOPLE_THRESHOLD (PENDING_CONFIRMATION)
    people_ok = people_ok_mask(products_df, PEOPLE_THRESHOLD)
    in_population = has_note & year_ok & people_ok
    products_df["in_population"] = in_population.astype(bool)

    notes_df = pd.DataFrame(notes_rows)
    if len(notes_df):
        notes_df["product_id"] = notes_df["product_id"].astype("Int64")
        notes_df["tier"] = notes_df["tier"].astype("category")
        notes_df["note_raw"] = notes_df["note_raw"].astype("string")

    accords_df = pd.DataFrame(accords_rows)
    if len(accords_df):
        accords_df["product_id"] = accords_df["product_id"].astype("Int64")
        accords_df["accord"] = accords_df["accord"].astype("string")
        accords_df["strength"] = accords_df["strength"].astype("Int64")

    ai_df = pd.DataFrame(ai_rows)
    if len(ai_df):
        ai_df["product_id"] = ai_df["product_id"].astype("Int64")
        ai_df["polarity"] = ai_df["polarity"].astype("category")
        ai_df["text"] = ai_df["text"].astype("string")
        ai_df["up_votes"] = ai_df["up_votes"].astype("Int64")
        ai_df["down_votes"] = ai_df["down_votes"].astype("Int64")

    products_df.to_parquet(OUT_DIR / "01_products.parquet", index=False)
    notes_df.to_parquet(OUT_DIR / "01_notes_long.parquet", index=False)
    accords_df.to_parquet(OUT_DIR / "01_accords_long.parquet", index=False)
    ai_df.to_parquet(OUT_DIR / "01_ai_summary_long.parquet", index=False)

    # --- diagnostics -----------------------------------------------------------
    n_no_note = int((~has_note).sum())
    n_no_year = int((has_note & ~year_ok).sum())
    n_no_people = int((has_note & year_ok & ~people_ok).sum())

    print("=" * 88)
    print("DIAGNOSTICS")
    print("=" * 88)
    print(f"rows read from {RAW_PATH}: {n_read:,} (bad JSON lines: {n_bad_json})")
    print()
    print("population funnel (has_note -> year_ok -> people_ok), threshold = "
          f"{PEOPLE_THRESHOLD} [CONFIRMED]:")
    print(f"  {n_read:,} total products")
    print(f"  - {n_no_note:,} dropped: no notes (tiered and flat both empty)")
    print(f"  - {n_no_year:,} dropped: year missing or outside [{YEAR_MIN}, {YEAR_MAX}]")
    print(f"  - {n_no_people:,} dropped: people missing or < {PEOPLE_THRESHOLD}")
    print(f"  = {int(in_population.sum()):,} products in_population")
    print()
    print("output shapes:")
    print(f"  01_products.parquet       {products_df.shape}")
    print(f"  01_notes_long.parquet     {notes_df.shape}")
    print(f"  01_accords_long.parquet   {accords_df.shape}")
    print(f"  01_ai_summary_long.parquet {ai_df.shape}")
    print()
    print("null counts (full corpus, all 131,930 products):")
    for col in ["year", "brand", "rating_avg", "people"]:
        n_null = int(products_df[col].isna().sum())
        print(f"  {col:<12} {n_null:,} null ({100 * n_null / len(products_df):.1f}%)")
    print()
    n_has_votes = int(products_df["has_votes"].sum())
    n_pop_has_votes = int((products_df["in_population"] & products_df["has_votes"]).sum())
    print(f"has_votes: {n_has_votes:,}/{len(products_df):,} products have >=1 real vote across "
          f"relation/seasons/daypart/community_gender ({100 * n_has_votes / len(products_df):.1f}%)")
    print(f"  of which in_population: {n_pop_has_votes:,}/{int(products_df['in_population'].sum()):,}")


if __name__ == "__main__":
    main()
