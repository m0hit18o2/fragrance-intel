# Fragrance Market Intelligence — Course Project

## What this project is

A solo Business Analytics / Data Mining course project. Goal: using public fine-fragrance data, identify 3 olfactive "white-space" territories where consumer demand signal is high relative to competitive supply, and present them as hypotheses for concept screening — NOT as launch decisions.

Scope is **fine fragrance only**. Do not introduce FMCG framing, shampoo/bodywash examples, or claims about mass-market transfer. That scope was deliberately cut.

This is an analytics project, not a software project. Optimize for correct, interpretable, reproducible analysis — not for architecture, abstraction, or production readiness. Plain scripts over classes. No web app, no API, no database.

Data
data/raw/perfumes.jsonl — 131,930 Fragrantica records, one JSON object per line. READ-ONLY.
Schema: see SCHEMA.md. Coverage stats: see DATA_DICTIONARY.md.
This is the ONLY data source. No review corpus, no Google Trends, no e-commerce dataset. Do not suggest adding sources.


## Pipeline structure

Numbered scripts in `src/`, each reads the previous checkpoint and writes the next:

1. `01_clean_fragrances.py` → `data/interim/01_fragrances_clean.parquet`
2. `02_taxonomy.py` → `data/interim/02_note_families.parquet` + `outputs/taxonomy_map.csv`
3. `03_supply_trend.py` → `data/interim/03_supply_by_family_year.parquet`
4. `04_demand_ratings.py` → `data/interim/04_demand_ratings.parquet`
5. `05_reviews_sentiment.py` → `data/interim/05_family_sentiment.parquet`
6. `06_whitespace_score.py` → `outputs/06_territory_scores.csv`
7. `07_figures.py` → `outputs/figures/`

Rules:
- Each script must be runnable standalone from its input checkpoint. NEVER write code that requires re-running from raw data.
- Every script ends by printing: input shape, output shape, rows dropped and why, null counts on key columns.
- Use asserts on row counts after merges/filters (e.g. `assert len(out) > 0.5 * len(inp)` unless a bigger drop is expected and documented).
- Random seeds fixed at 42 everywhere (numpy, sklearn).

Fixed analytical decisions — implement exactly, do not substitute

If you think one of these is wrong, say so and wait for my decision. Never silently change one.

0. Analysis population (applies to EVERYTHING downstream)

Script 01 defines one population and every later script uses it. Filters:

people >= 50 (median across the corpus is only 15 — most products are noise)
has at least one note (notes.tiered any tier, or notes.flat)
year present and between 1990 and 2025 inclusive
exclude year >= 2026 — dump captured mid-2026, so 2026 is a partial year and 2027 is a pre-announcement

Script 01 must print a table of surviving row counts at people thresholds of 20 / 50 / 100 / 200, and the % of each that also has ai_summary.pros, BEFORE I commit to 50. Show me that table and stop.

1. Note normalisation (new, required, runs before the taxonomy)

Raw vocabulary is 2,520 tokens with duplicates and non-notes. Build outputs/note_normalisation.csv (raw_token, canonical, action) by:

lowercase, strip, collapse whitespace, normalise hyphens to spaces (lily-of-the-valley -> lily of the valley)
strip parenthetical glosses (agarwood (oud) -> agarwood), then map known synonym pairs to one canonical form
DROP placeholder tokens that are categories not notes: anything matching ^(woody|floral|green|fruity|spicy|woodsy|sea|citrus|white|aromatic|powdery|earthy|animalic|balsamic|aquatic|herbal|smoky|marine|ozonic|salty|nutty|lactonic|tropical|mossy|soapy|conifer|metallic|sour|bitter|mineral|savory|terpenic|oily) notes?$, plus citruses, spices, white flowers
drop tokens appearing on fewer than 30 products in the analysis population

Propose the synonym pairs by string similarity and frequency; write them to the CSV with your suggested canonical form. I review and edit that CSV before script 02 uses it. Do not hardcode a mapping from your own perfumery knowledge.

2. Taxonomy

Note co-occurrence matrix (canonical notes x notes, counted over products in the analysis population) -> PPMI weighting -> TruncatedSVD to 50 dims -> KMeans, k=15 (also fit 12 and 18 for comparison; I pick final k) -> I hand-label clusters into named families. Save cluster assignments to outputs/taxonomy_map.csv for my review. Random seed 42.

Validation (required): for each learned cluster, report the most over-represented Fragrantica accords among products loading on it. Accords are 97.9% covered and a controlled vocabulary of 92 — they are the external check that the learned clusters are real. Save to outputs/taxonomy_validation.csv.

A product's family membership = share of its canonical notes falling in each family (fractional, not single-label).

3. Demand — three signals, computed separately, never averaged into one number before I see them
Approval: Bayesian-shrunk rating (v*R + m*C)/(v+m), v = people, R = rating.average, C = global mean, m = median people in the analysis population.
Desire: relation.want / (relation.have + relation.had + relation.want) — unmet intent.
Adoption: log1p(relation.have).
Value perception: price_value.average.

Do NOT use popularity.magnitude — the schema states it is ~1.47*(have+had+want), so it duplicates relation.

Brand-demeaning: subtract the brand mean from each signal, but ONLY within brands having >= 5 products in the analysis population (7,876 brands, most with one product; demeaning a singleton brand yields exactly zero). Products in smaller brands keep raw values and get a brand_demeaned boolean flag. Report family demand both ways.

4. Supply and momentum
Supply = share of launches per family per year, never raw counts. Absolute counts rise ~9x from 2000 to 2025 because Fragrantica indexes recent releases more completely — that is coverage bias, not market growth. Say so in any output.
Momentum = (mean family share 2020-2024) - (mean family share 2015-2019). Fixed 5-year blocks; do not use trailing windows against the ragged 2025/2026 edge.
5. Sentiment — from ai_summary, no external corpus

Present on 9.4% of the corpus (12,432 products), concentrated in popular products. Restrict to analysis-population products that have it, define that subpopulation explicitly, and report its size and how it differs from the full population (mean people, mean year, top brands).

Method: for each product, map its ai_summary.pros and .cons text against accord/family-level vocabulary (fresh, sweet, woody, citrus, gourmand, creamy, warm, floral, powdery...), not fine-grained note names — the pros text is dominated by performance and occasion words, so family-level terms are what actually appear. Weight each pro/con item by up_votes - down_votes (floor at 0). Per family: net positive mention weight vs net negative.

No VADER, no nltk, no transformers. The pros/cons split IS the polarity label.

6. Seasonality — native

From seasons vote counts (100% coverage). Per family, share of votes by season, weighted by product family membership. This replaces Google Trends entirely.

7. White-space score

Per family, z-scored across families: score = z(approval) + z(desire) + z(sentiment_net) - z(supply_share) + 0.5*z(momentum) Weights are fixed. Do not tune them. Report every component alongside the total so the ranking is traceable.

8. Language discipline

"associated with", never "drives" or "causes". Ratings are enthusiast approval, not sales. Every output that ranks families must carry the framing: hypotheses for concept screening, not launch decisions.

## Environment

- Python 3.11+, venv at `.venv`
- Allowed libraries ONLY: pandas, numpy, scipy, scikit-learn, pyarrow, matplotlib, vaderSentiment, nltk (punkt only). Ask before adding anything else. Never install torch/transformers — not needed and too slow to download.
- No network calls in any pipeline script. All data is local files.
- Plots: matplotlib only, one chart per figure, readable labels, saved as PNG at 200 dpi.

## Workflow rules

- Work on ONE numbered script at a time. After it runs clean and prints its shape report, stop and show me the printed diagnostics before moving to the next.
- When a task fails twice with the same class of error, stop and explain the problem instead of trying a third variation.
- Do not create a Jupyter notebook until all scripts work; the final notebook is assembled from working scripts at the end.
- Do not write the report or slides text; I write those. You may generate tables/figures I ask for.
- Keep `PROGRESS.md` updated: after finishing each script, append one line — what was produced, row counts, anything surprising in the data.

## Definition of done (for the code)

All 7 scripts run end-to-end from checkpoints; `outputs/06_territory_scores.csv` ranks families with all component scores visible; figures exist for (a) 2-D olfactive map with labeled clusters, (b) family launch-share over time, (c) demand-vs-supply scatter; every number in these outputs is traceable to a script.