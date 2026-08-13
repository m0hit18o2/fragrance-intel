# Environment

`requirements.txt` is a full `pip freeze` of the project venv (74 packages,
including transitive dependencies). This file lists only what was added
beyond CLAUDE.md's original allowed list (pandas, numpy, scipy,
scikit-learn, pyarrow, matplotlib, vaderSentiment, nltk) and why.

| package | why it was needed |
|---|---|
| `kneed` | `03b_cluster_selection.py` — `KneeLocator` on the inertia curve, requested by name for the k-selection diagnostic. |
| `umap-learn` | `03c_nonlinear_clustering.py` — UMAP→HDBSCAN, the standard manifold-learning pipeline requested for checking whether the note embedding has structure raw HDBSCAN misses. |
| `minisom` | `03c_nonlinear_clustering.py` — the self-organising map (6x6 grid) requested as one of the five clustering-evidence checks. |
| `gensim` | `03c_nonlinear_clustering.py` — skip-gram Word2Vec, requested as the neural-embedding comparison against the PPMI+SVD partition. |
| `networkx` | `03c_nonlinear_clustering.py` — Louvain community detection on the note co-occurrence graph, requested by name (`networkx`'s own `louvain_communities`, not the separate `python-louvain` package). |
| `nbformat` | Building `notebooks/results.ipynb` programmatically (constructing/reading/writing the `.ipynb` cell structure) — no notebook tooling was in the original allowed list at all, since one hadn't been needed yet. |
| `ipykernel` | Registering a venv-pinned Jupyter kernel (`fragrance-intel`) so the notebook resolves `pandas` etc. regardless of how it's opened, rather than relying on a bare `python` on `PATH` (see PROGRESS.md's kernel-fix entry). |
| `nbclient` | Actually executing the notebook end-to-end (restart-and-run-all) to verify it, rather than trusting untested cells. |

## Removed

| package | why it was removed |
|---|---|
| `hdbscan` | **Removed 2026-08-13.** Was present but unused: installed during initial exploration for `03b_cluster_selection.py` (same timestamp as `kneed`) while checking whether to use the standalone package or scikit-learn's built-in `sklearn.cluster.HDBSCAN`. The built-in was what got used in both `03b` and `03c` (confirmed via `grep -rn "import hdbscan" src/` — zero hits). Confirmed via `pip show hdbscan` that nothing else in the environment depends on it (`Required-by:` empty) before removing with `pip uninstall hdbscan`. Re-verified afterward that `sklearn.cluster.HDBSCAN` and every other project import still work and that `03c_nonlinear_clustering.py` still compiles. |

While cleaning this up, also removed a handful of packages (`build`, `installer`, `pyproject_hooks`, `tomli`, `tomli_w`, `truststore`, `nab-index`, `nab-python`, `nab-resolver`, `pipdeptree`) that turned out to be pip's own packaging/build tooling and a dependency-tree inspector used transiently to verify the `hdbscan` removal — not project dependencies, never imported by anything in `src/`.

**Not a project dependency**: `jsdom` (and its own transitive npm packages) was installed under `/tmp` via `npm install --prefix /tmp/npmtest` solely to test `outputs/app/index.html`'s JavaScript in Node (search, filtering, drill-down) before handing it off. It is a Node/npm package, not a Python one — it was never installed into the venv, does not appear in `requirements.txt`, and the `/tmp` install was deleted after testing. Nothing in the deliverables depends on it.
