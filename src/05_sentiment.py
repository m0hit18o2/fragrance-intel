"""
05_sentiment.py — family-level sentiment from ai_summary.pros/cons, no
external corpus (CLAUDE.md section 5).

Subpopulation: in_population AND has_ai_summary. Maps each pro/con item's
text against family-level vocabulary (each cluster's top-lift accords from
03_taxonomy.py, split into words, plus obvious synonyms) -- not fine-grained
note names, since the pros/cons text is dominated by performance/occasion
words and family-level terms are what actually appear. Each item is weighted
by max(up_votes - down_votes, 0). The pros/cons split IS the polarity label
(no VADER/nltk/transformers).

Output: data/interim/05_family_sentiment.parquet
Run standalone: python src/05_sentiment.py
(inputs: data/interim/01_products.parquet, 01_ai_summary_long.parquet,
outputs/taxonomy_validation.csv)
"""
import re
from pathlib import Path

import pandas as pd

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")

# Obvious morphological synonyms for the accord words that show up in
# taxonomy_validation.csv's top-lift lists -- plurals / noun-adjective pairs,
# not perfumery family judgments. Every base word maps at least to itself.
SYNONYMS = {
    "sour": ["sour", "sourness"],
    "aquatic": ["aquatic", "aqua"],
    "tropical": ["tropical", "tropics"],
    "ozonic": ["ozonic", "ozone"],
    "floral": ["floral", "flowery", "flower", "flowers"],
    "camphor": ["camphor", "camphorous"],
    "cannabis": ["cannabis", "weed", "hemp"],
    "bitter": ["bitter", "bitterness"],
    "conifer": ["conifer", "pine", "pines"],
    "herbal": ["herbal", "herbaceous", "herbs", "herb"],
    "balsamic": ["balsamic", "balsam", "balsamy"],
    "rum": ["rum"],
    "mineral": ["mineral", "minerally", "minerals"],
    "vanilla": ["vanilla", "vanillic"],
    "almond": ["almond", "almondy"],
    "tuberose": ["tuberose"],
    "earthy": ["earthy", "earth", "earthiness"],
    "iris": ["iris", "orris"],
    "patchouli": ["patchouli"],
    "soapy": ["soapy", "soap", "soapiness"],
    "lavender": ["lavender", "lavandin"],
    "violet": ["violet", "violets"],
    "cherry": ["cherry", "cherries"],
    "lactonic": ["lactonic", "creamy", "milky", "cream"],
    "nutty": ["nutty", "nut", "nuts"],
    "caramel": ["caramel", "caramelized", "caramelised"],
    "rose": ["rose", "rosy", "roses"],
    "musky": ["musky", "musk", "musks"],
    "aldehydic": ["aldehydic", "aldehyde", "aldehydes"],
    "oud": ["oud", "agarwood"],
    "animalic": ["animalic", "animal", "animalistic"],
    "smoky": ["smoky", "smoke", "smokey", "smokiness"],
    "leather": ["leather", "leathery"],
    "whiskey": ["whiskey", "whisky", "bourbon"],
    "cacao": ["cacao", "cocoa"],
    "coffee": ["coffee", "espresso"],
    "honey": ["honey", "honeyed"],
    "terpenic": ["terpenic", "terpene", "terpenes"],
    "savory": ["savory", "savoury"],
    "chocolate": ["chocolate", "chocolatey", "chocolaty"],
    "metallic": ["metallic", "metal", "metals"],
    "coconut": ["coconut", "coconutty"],
    "salty": ["salty", "salt", "saline"],
    "anis": ["anis", "anise", "licorice", "liquorice"],
    "tobacco": ["tobacco"],
    "yellow": ["yellow"],
    "warm": ["warm", "warmth"],
    "spicy": ["spicy", "spice", "spices"],
    "white": ["white"],
    "fresh": ["fresh", "freshness"],
    "sweet": ["sweet", "sweetness"],
    "woody": ["woody", "wood", "woods"],
    "citrus": ["citrus", "citrusy", "citric"],
    "powdery": ["powdery", "powder"],
    "aromatic": ["aromatic", "aroma"],
    "fruity": ["fruity", "fruit", "fruits"],
    "amber": ["amber", "ambery"],
    "green": ["green", "greenery"],
    "gourmand": ["gourmand", "gourmet"],
}


def cluster_terms(accords):
    """accord phrases -> deduped {base_word: [synonym forms]} flattened to a
    single term list for this cluster (splits multi-word accords like
    "yellow floral" -> yellow, floral; "fresh spicy" -> fresh, spicy)."""
    words = set()
    for phrase in accords:
        for w in phrase.split(" "):
            words.add(w)
    terms = set()
    for w in words:
        terms.update(SYNONYMS.get(w, [w]))
    return sorted(terms)


def compile_cluster_regex(terms):
    escaped = [re.escape(t) for t in terms]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def main():
    products = pd.read_parquet(
        IN_DIR / "01_products.parquet",
        columns=["id", "in_population", "has_ai_summary", "people", "year", "rating_avg"],
    )
    ai = pd.read_parquet(IN_DIR / "01_ai_summary_long.parquet")
    validation = pd.read_csv(OUT_DIR / "taxonomy_validation.csv")

    pop = products[products["in_population"]].copy()
    subpop = pop[pop["has_ai_summary"]].copy()

    print(f"in_population: {len(pop):,}")
    print(f"subpopulation (in_population AND has_ai_summary): {len(subpop):,} "
          f"({100 * len(subpop) / len(pop):.1f}% of in_population)")
    print()
    print("subpopulation vs full in_population:")
    for col, label in [("people", "mean people"), ("year", "mean year"), ("rating_avg", "mean rating")]:
        print(f"  {label:<14} subpop={subpop[col].mean():.2f}   full_population={pop[col].mean():.2f}")
    print()

    # --- family-level term lists, from taxonomy_validation.csv's top accords -
    cluster_name = validation[["cluster", "cluster_name"]].drop_duplicates().set_index("cluster")["cluster_name"]
    terms_by_cluster = {}
    regex_by_cluster = {}
    for c, grp in validation.groupby("cluster"):
        terms = cluster_terms(grp["accord"].tolist())
        terms_by_cluster[c] = terms
        regex_by_cluster[c] = compile_cluster_regex(terms)

    print("family term lists (top accords, split + synonym-expanded):")
    for c in sorted(terms_by_cluster):
        print(f"  cluster {c:>2} ({cluster_name[c]}): {terms_by_cluster[c]}")
    print()

    # --- restrict ai items to subpopulation -------------------------------------
    subpop_ids = set(subpop["id"])
    ai_sub = ai[ai["product_id"].isin(subpop_ids)].copy()
    ai_sub["weight"] = (ai_sub["up_votes"].fillna(0) - ai_sub["down_votes"].fillna(0)).clip(lower=0)
    ai_sub["text"] = ai_sub["text"].fillna("")
    print(f"ai_summary pro/con items in subpopulation: {len(ai_sub):,} "
          f"(pros={int((ai_sub['polarity'] == 'pro').sum()):,}, "
          f"cons={int((ai_sub['polarity'] == 'con').sum()):,})")

    # --- match each item against each cluster's regex, accumulate weight -------
    pos_weight = {c: 0.0 for c in terms_by_cluster}
    neg_weight = {c: 0.0 for c in terms_by_cluster}
    pos_n = {c: 0 for c in terms_by_cluster}
    neg_n = {c: 0 for c in terms_by_cluster}

    for text, polarity, weight in zip(ai_sub["text"], ai_sub["polarity"], ai_sub["weight"]):
        for c, rx in regex_by_cluster.items():
            if rx.search(text):
                if polarity == "pro":
                    pos_weight[c] += weight
                    pos_n[c] += 1
                else:
                    neg_weight[c] += weight
                    neg_n[c] += 1

    rows = []
    for c in sorted(terms_by_cluster):
        rows.append({
            "cluster": c,
            "cluster_name": cluster_name[c],
            "n_pro_items_matched": pos_n[c],
            "n_con_items_matched": neg_n[c],
            "net_positive_weight": pos_weight[c],
            "net_negative_weight": neg_weight[c],
            "net_score": pos_weight[c] - neg_weight[c],
        })
    family_sentiment = pd.DataFrame(rows)
    family_sentiment.to_parquet(IN_DIR / "05_family_sentiment.parquet", index=False)

    # --- diagnostics -------------------------------------------------------------
    print()
    print("=" * 100)
    print("DIAGNOSTICS")
    print("=" * 100)
    print(f"output shape: 05_family_sentiment.parquet {family_sentiment.shape}")
    print()
    with pd.option_context("display.width", 140, "display.float_format", "{:.1f}".format):
        print(family_sentiment.sort_values("net_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
