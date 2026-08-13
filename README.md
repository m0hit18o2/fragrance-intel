# Fragrance Market Intelligence — Olfactive White-Space Analysis

**Group 5**

| Name | Roll Number |
|---|---|
| Mohitabinav M | 0120/62 |
| Raijas KP | 0290/62 |
| Govind S | 0347/62 |
| Sumant Saurav | 0392/62 |
| Sushma K | 0393/62 |

## Deployed app

Not deployed. The app is included in this submission as
`outputs/app/index.html` — a self-contained file that opens directly in
any browser by double-clicking it, no server or internet connection
required. [APP LINK]

## YouTube link

https://youtu.be/GqPExsjt_24

## How to run the code

1. Install Python 3.12, create a virtual environment, and install from
   `requirements.txt` (see Environment below).
2. Place the raw data at `data/raw/perfumes.jsonl` (see Data below —
   we've included a 5,000-record sample at `data/sample/perfumes_sample.jsonl`
   since the full file is too large to submit).
3. Run every script in `src/` from the repository root, in numeric/
   alphabetical order (`01_clean.py` first, `11_param_optimisation.py`
   last). The full run order with what each script produces is in the
   "Run order" section below.
4. Open `notebooks/results.ipynb` and `outputs/app/index.html` to see
   the results (see "Opening the notebook and the app" below).

## What we built

We analysed a dump of 131,930 Fragrantica fine-fragrance records to find
olfactive "white-space" territories — note families where enthusiast
demand looks high relative to how many products currently compete in
that space — and present them as hypotheses for concept screening, not
launch decisions. Along the way we built a normalised taxonomy of ~500
canonical notes into 15 families (grouped by which notes actually
co-occur in real products, not by fragrance-industry convention),
computed demand/supply/sentiment/seasonality signals per family, and
built a "similar products" recommender evaluated against the site's own
community-voted "reminds me of" data. The two end-user-facing outputs
are `outputs/06_territory_scores.csv` (the ranked list of families with
every scoring component visible) and `outputs/app/index.html` (a
self-contained, offline, two-tab tool for exploring both the note
taxonomy and product similarity interactively). `notebooks/results.ipynb`
is a narrative walkthrough that loads and displays everything the
pipeline produced, without recomputing any of it.

## Environment

- Python 3.12 (we built and verified this against 3.12.3; other 3.11+
  versions will likely work but we haven't tested them).
- A virtual environment, then install from the pinned lockfile:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full `pip freeze` (74 packages including
transitive dependencies) captured from the environment we built this in.
See `ENVIRONMENT.md` for which packages we added beyond a minimal
analysis stack and why we needed each one.

## Data

The full raw dump belongs at exactly:

```
data/raw/perfumes.jsonl
```

One JSON object per line, 131,930 records — see `SCHEMA.md` for the
exact record structure and `DATA_DICTIONARY.md` for field-coverage
statistics. **We have not included this file in the submission — at
~500MB it's too large to upload.**
https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes

In its place, we've included `data/sample/perfumes_sample.jsonl` — 5,000
records sampled (seed 42) from the raw file, biased toward products in
our analysis population (4,000 of the 5,000 are in-population products,
1,000 are not) so the sample is actually usable for exercising the
pipeline's logic end-to-end, not just a random slice that happens to
mostly miss the population filters. It is not a substitute for the full
file for reproducing our actual numbers — some steps (note-taxonomy
co-occurrence counts, popularity baselines, etc.) need the full corpus
to match the committed outputs — but it's enough to run every script
without errors and sanity-check the pipeline's structure.

## Run order

Run every script from the repository root, in this order. Each one reads
only from checkpoints earlier scripts already wrote (or from `data/raw/`
directly, for `01_clean.py` and `01c_extract_similar.py`, which stream it
line by line rather than loading it whole) — never re-run a later script
without having run everything before it at least once. Runtimes below
are rough, on a single laptop-class machine with no GPU;
`08_recommender.py` and `11_param_optimisation.py` dominate the total.

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

Two things in this repository were written or edited by hand and are
**not reproducible by re-running the pipeline from raw data**. Anyone
re-running the pipeline should keep these as-is rather than expect a
script to regenerate them:

- **`outputs/cluster_names_final.csv`** — the 15 families' short display
  names, which we hand-wrote after reviewing the auto-generated long
  names in `outputs/cluster_names.csv` (which `06_whitespace.py` does
  regenerate automatically, from each family's top-8-notes-by-frequency —
  that one *is* a pipeline output). `cluster_names_final.csv` is read by
  `07_figures.py`, `03c_nonlinear_clustering.py`, `10_build_app.py`, and
  the notebook. If it's missing, those scripts will fail or use whatever
  stale copy happens to exist.
- **The manual override list inside `02_normalise_notes.py`** — the
  `OVERRIDE_EXCLUDE`, `OVERRIDE_FORCE_MERGE`, and `OVERRIDE_CANONICAL`
  dictionaries near the top of the file (e.g. un-merging "ambergris" from
  "amber", promoting "virginia cedar" → "cedar"). These are hardcoded in
  the script, so re-running it always reproduces the same result — but
  the entries themselves are our corrections to what the automated
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
(fragrance-intel .venv)" kernel if it isn't chosen automatically. All
cell outputs are saved in the submitted file, so it can also just be read
top to bottom without running anything.

**App**: `outputs/app/index.html` is a single self-contained file — no
build step, no server. Double-click it, or open it directly in any
browser. It's about 10MB, so it can take a few seconds to load on first
open (parsing the embedded data payload) — this is normal, not a hang.
It works fully offline (e.g. from a USB stick); nothing it does depends
on network access.

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

We last verified this 2026-08-13, against the pipeline state submitted
here (rebuilt `08_recommender.py` with the reconciled hybrid definition,
`04_demand.py` also writing `04_product_demand.parquet`, the trimmed
74-package `requirements.txt`, and the two-tab `10_build_app.py`). We
built a fresh virtual environment from the current `requirements.txt`,
and re-ran the full pipeline — all 15 scripts, in order — end-to-end from
raw data in an isolated directory (raw data symlinked,
`cluster_names_final.csv` copied in as the one human-authored
non-pipeline input), then diffed the result against the committed
outputs. We verified these identical:

- `outputs/06_territory_scores.csv` — identical to 3 decimal places on
  every column (0/255 numeric cells differ).
- `outputs/taxonomy_map.csv` — Adjusted Rand Index of 1.000000 between the
  committed and freshly-computed cluster assignments (492/492 notes), and
  in fact exact label equality too, not just partition equality.
- `outputs/09_evaluation.csv` — identical to 4 decimal places on every
  algorithm and metric (0/104 numeric cells differ).
- `outputs/11_test_results.csv` — identical to 4 decimal places on every
  algorithm and metric (0/143 numeric cells differ).
- `outputs/note_normalisation.csv` — byte-identical (MD5 match).
- `outputs/app/index.html` — same file size (10,755,196 bytes) in both,
  and the embedded JSON payload parses cleanly in both; it also matched
  byte-for-byte (MD5 match) this run, though that isn't guaranteed on
  every run since dict ordering inside the payload isn't pinned.

No divergence was found on this check. An earlier check (against an
older pipeline state, before we reconciled the hybrid representation's
definition) had found one cosmetic divergence, which we've since fixed:
a single row in `outputs/note_normalisation.csv` — the note `calone`'s
REVIEW annotation — cited a different one of two exactly-tied
fuzzy-match candidates (`calypsone` vs. `cascalone`, both similarity
0.80) depending on the process's Python string-hash seed, which is
randomised by default per process. This never affected any
`canonical`/`action` column, only which example was named in a
human-facing note, and it never affected any number anywhere downstream.
We fixed it with an explicit alphabetical tie-break in
`02_normalise_notes.py` (not by relying on setting `PYTHONHASHSEED`),
verified at the time by running the script in two separate processes
with confirmed different hash seeds and checking the output was
byte-identical both times — and this check's byte-identical
`note_normalisation.csv` result confirms that fix has held under the
current pipeline state too.
