# Fragrance Market Intelligence

## What this is

This project analyses a dump of 131,930 Fragrantica fine-fragrance records
to find olfactive "white-space" territories — note families where
enthusiast demand looks high relative to how many products currently
compete in that space — and presents them as hypotheses for concept
screening, not launch decisions. Along the way it builds a normalised
taxonomy of ~500 canonical notes into 15 families (grouped by which notes
actually co-occur in real products, not by fragrance-industry convention),
computes demand/supply/sentiment/seasonality signals per family, and
builds a "similar products" recommender evaluated against the site's own
community-voted "reminds me of" data. The two end-user-facing outputs are
`outputs/06_territory_scores.csv` (the ranked list of families with every
scoring component visible) and `outputs/app/index.html` (a self-contained,
offline, two-tab tool for exploring both the note taxonomy and product
similarity interactively). `notebooks/results.ipynb` is a narrative
walkthrough that loads and displays everything the pipeline produced,
without recomputing any of it.

## Requirements

- Python 3.12 (this was built and verified against 3.12.3; other 3.11+
  versions will likely work but haven't been tested).
- A virtual environment, then install from the pinned lockfile:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full `pip freeze` (74 packages including
transitive dependencies) captured from the environment this was built in.
See `ENVIRONMENT.md` for which packages were added beyond a minimal
analysis stack and why each was needed.

## Raw data

Place the dump at exactly:

```
data/raw/perfumes.jsonl
```

One JSON object per line, 131,930 records — see `SCHEMA.md` for the exact
record structure and `DATA_DICTIONARY.md` for field-coverage statistics.
This repository does not include the raw dump (it's gitignored, and it's
~500MB) and this README does not have a download link for it — obtain it
from whoever supplied it to you. `data/raw/` and `data/interim/` are both
gitignored; only `outputs/` and `notebooks/` are committed.

## Running the pipeline

Run every script from the repository root, in this order. Each one reads
only from checkpoints earlier scripts already wrote (or from `data/raw/`
directly, for `01_clean.py` and `01c_extract_similar.py`, which stream it
line by line rather than loading it whole) — never re-run a later script
without having run everything before it at least once. Runtimes are rough,
on a single laptop-class machine with no GPU; `08_recommender.py` and
`11_param_optimisation.py` dominate the total.

| # | script | produces | rough runtime |
|---|---|---|---|
| 1 | `01_clean.py` | Streams the raw JSONL into 4 parquet checkpoints (products, notes, accords, AI-summary pros/cons) and defines the 32,499-product analysis population (≥50 raters, ≥1 note, launch year 1990–2025). | 30–60s |
| 2 | `02_normalise_notes.py` | Collapses ~2,000 raw note-name spelling/naming variants into 492 canonical notes, via fuzzy matching plus a small hand-reviewed override list. Writes `outputs/note_normalisation.csv`. | ~5s |
| 3 | `03_taxonomy.py` | Note co-occurrence → PPMI → SVD (50-d) → KMeans (k=15): the production note taxonomy. Writes `outputs/taxonomy_map.csv`, `outputs/taxonomy_validation.csv`, `data/interim/03_product_family.parquet`. | ~5s |
| 4 | `03b_cluster_selection.py` | Evidence for the k=15 choice (does not change it): k-sweep, bootstrap stability, external validity against accords, alternative algorithms. | ~50s |
| 5 | `03c_nonlinear_clustering.py` | Five more independent checks on whether k=15 finds real discrete structure: hierarchical clustering, UMAP+HDBSCAN, skip-gram embedding, self-organising map, Louvain community detection. | ~30s |
| 6 | `04_demand.py` | Approval (Bayesian-shrunk rating), desire (want-ratio), adoption, and value-perception signals, per product and per family, brand-demeaned. | ~2s |
| 7 | `05_sentiment.py` | Maps each product's AI-summary pros/cons text to family-level vocabulary, weighted by net agreement votes. | ~10s |
| 8 | `05b_seasonality.py` | Season-vote share per family. | ~2s |
| 9 | `06_whitespace.py` | The composite whitespace score (demand + sentiment − supply + momentum, brand-demeaned; see `METHODS_NOTES.md`). Writes `outputs/06_territory_scores.csv`. | ~2s |
| 10 | `07_figures.py` | Three whitespace-analysis figures: note map, family launch-share over time, demand vs. supply. | ~10s |
| 11 | `01c_extract_similar.py` | Extracts the "reminds me of" / "also liked" carousels as recommender ground truth. | ~10s |
| 12 | `08_recommender.py` | Builds five product-similarity representations (note embeddings, TF-IDF, family membership, accords, a tuned hybrid) and their top-50 nearest-neighbour lists. | 1.5–2 min |
| 13 | `09_recommender_eval.py` | Evaluates all five representations plus three baselines (random, popularity, same-brand) against the community ground truth. Writes `outputs/09_evaluation.csv`. | ~10s |
| 14 | `10_build_app.py` | Builds the self-contained two-tab HTML app. Writes `outputs/app/index.html`. | ~15–20s |
| 15 | `11_param_optimisation.py` | Leakage-safe hyperparameter tuning: splits ground-truth queries 60/20/20 by product, tunes on validation only, reports a final test-set comparison. | 2–3 min |

That's 15 scripts. `src/10_app_template.html` is not a script — it's the
HTML/CSS/JS template `10_build_app.py` fills in; it isn't run on its own.

## Files that are human-authored inputs, not pipeline outputs

Two things in this repository were written or edited by a person and are
**not reproducible by re-running the pipeline from raw data**. Anyone
re-running the pipeline should keep these as-is rather than expect a
script to regenerate them:

- **`outputs/cluster_names_final.csv`** — the 15 families' short display
  names, hand-written after reviewing the auto-generated long names in
  `outputs/cluster_names.csv` (which `06_whitespace.py` does regenerate
  automatically, from each family's top-8-notes-by-frequency — that one
  *is* a pipeline output). `cluster_names_final.csv` is read by
  `07_figures.py`, `03c_nonlinear_clustering.py`, `10_build_app.py`, and
  the notebook. If it's missing, those scripts will fail or use whatever
  stale copy happens to exist.
- **The manual override list inside `02_normalise_notes.py`** — the
  `OVERRIDE_EXCLUDE`, `OVERRIDE_FORCE_MERGE`, and `OVERRIDE_CANONICAL`
  dictionaries near the top of the file (e.g. un-merging "ambergris" from
  "amber", promoting "virginia cedar" → "cedar"). These are hardcoded in
  the script, so re-running it always reproduces the same result — but
  the entries themselves are a person's corrections to what the automated
  fuzzy-matching proposed, not something the algorithm derived on its
  own. If the underlying note vocabulary changes (e.g. the raw dump is
  updated), these should be reviewed again, not assumed to still apply.

## Opening the notebook and the app

**Notebook**: `notebooks/results.ipynb` loads only saved outputs — it does
no computation of its own, so it only needs the pipeline to have been run
at least once. Register the pinned kernel once, so the notebook resolves
`pandas` etc. regardless of how it's opened:

```
python -m ipykernel install --user --name=fragrance-intel --display-name="Python (fragrance-intel .venv)"
```

Then open it in Jupyter, JupyterLab, or VS Code, and select the "Python
(fragrance-intel .venv)" kernel if it isn't chosen automatically.

**App**: `outputs/app/index.html` is a single self-contained file — no
build step, no server. Double-click it, or open it directly in any
browser. It works offline (e.g. from a USB stick); nothing it does
depends on network access.

## Outputs guide — which file answers which question

| question | file(s) |
|---|---|
| Which families look like whitespace territories? | `outputs/06_territory_scores.csv`, figure `outputs/figures/c_demand_vs_supply.png` |
| What are the 15 families, and what defines each one? | `outputs/taxonomy_map.csv`, `outputs/taxonomy_validation.csv`, `outputs/cluster_names_final.csv`, figure `a_note_map.png` |
| Is k=15 actually the right number of families? | `outputs/cluster_algorithm_comparison.csv`, `outputs/03c_summary.csv`, `outputs/hierarchical_k5_superfamilies.csv`, figures `d`–`g`, `j` |
| How was raw note text cleaned up? | `outputs/note_normalisation.csv` |
| How is supply changing over time, and is that a real trend or a coverage artefact? | `outputs/figures/b_launch_share_over_time.png` (see the caption — raw counts rise ~9x 2000–2025 from Fragrantica indexing recent releases more completely, not from real growth) |
| How good is the "similar products" recommender, and against what baseline? | `outputs/09_evaluation.csv` |
| Did tuning the recommender actually help, or was it noise? | `outputs/11_test_results.csv`, `outputs/11_param_search.csv`, figure `k_param_search.png` |
| Explore a specific note's family, partners, reception, and example products | `outputs/app/index.html`, Note Explorer tab |
| Find products similar to a specific product, and see why | `outputs/app/index.html`, Product Similarity tab |
| One narrative walkthrough of all of the above | `notebooks/results.ipynb` |
| Intermediate checkpoints (not meant to be read directly — later scripts consume these) | `data/interim/*.parquet`, `data/interim/*.npz` |

## Reproducibility

A fresh venv was built from `requirements.txt`, and the full pipeline was
re-run end-to-end from raw data in an isolated directory, then diffed
against the committed outputs. Verified identical:

- `outputs/06_territory_scores.csv` — identical to 3 decimal places on
  every column.
- `outputs/taxonomy_map.csv` — Adjusted Rand Index of 1.000000 between the
  committed and freshly-computed cluster assignments (492/492 notes), and
  in fact exact label equality too, not just partition equality.
- `outputs/09_evaluation.csv` — identical to 4 decimal places on every
  algorithm and metric.

One cosmetic divergence was found (beyond the three files above, which
were the ones specifically checked) and has since been fixed: a single
row in `outputs/note_normalisation.csv` — the note `calone`'s REVIEW
annotation — cited a different one of two exactly-tied fuzzy-match
candidates (`calypsone` vs. `cascalone`, both similarity 0.80) depending
on the process's Python string-hash seed, which is randomised by default
per process. This never affected any `canonical`/`action` column, only
which example was named in a human-facing note, and it never affected any
number anywhere downstream. The fix (an explicit alphabetical tie-break in
`02_normalise_notes.py`, not reliance on setting `PYTHONHASHSEED`) was
verified by running the script in two separate processes with confirmed
different hash seeds and checking the output was byte-identical both times.
