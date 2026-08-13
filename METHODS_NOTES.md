# Methods notes

Modelling choices made during the build that are not among CLAUDE.md's
fixed decisions, and so were not handed down — they were decided along
the way and are recorded here so they don't stay ambiguous. Each is also
noted as a comment at its point of use in the code.

## Brand-demeaning in the composite whitespace score — CONFIRMED

`06_whitespace.py`'s composite score (`score = z(approval) + z(desire) +
z(sentiment_polarity) - z(supply_share) + 0.5*z(momentum)`) uses the
**brand-demeaned** approval and desire signals from `04_demand.py`, not
the raw ones.

**Why**: raw approval/desire are confounded by brand-prestige halo — a
product from a heritage house scores well for being that house's, not
because of its family. A white-space read is asking "does this *family*
attract enthusiast preference," and answering that requires netting out
which specific brands happen to dominate a family, not just measuring
family averages as-is.

**What this means in practice**:
- The composite score, and therefore the ranking in
  `outputs/06_territory_scores.csv` and everywhere that ranking is used
  (the notebook, the app), is always computed from demeaned values.
- Raw values are still written to `06_territory_scores.csv` and shown in
  the notebook/app — but only where an **absolute level** is being
  described (e.g. "this family's raw mean approval is 3.97 out of 5"),
  never fed into the score itself.
- `04_demand.py`'s per-product brand-demeaning is unchanged by this note:
  it already only demeans within brands with ≥5 in-population products
  (7,876+ brands corpuswide, most with a single product — demeaning a
  singleton brand is definitionally zero), flagging smaller-brand
  products with `brand_demeaned=False` and leaving their raw values in
  place. This note is about which of the two versions the *whitespace
  score* consumes, not about how demeaning itself is computed.

This does not change any number already reported — it makes explicit
what the code already did, since the code comment in `06_whitespace.py`
previously described this as a choice still open for review.

## Hybrid representation, tuned vs. default alpha

`08_recommender.py`'s "hybrid" representation and `11_param_optimisation.py`'s
alpha-tuning were reconciled to the same definition (see `PROGRESS.md`
for the full account: they used to be two different objects — z-scored
note_svd+accord concatenation vs. a note_tfidf+accord score blend — being
compared as if interchangeable, which was invalid). The resolved
definition is a score-level blend:

    alpha * cosine(note_tfidf) + (1 - alpha) * cosine(accord)

`08_recommender.py` builds this with `alpha=0.5` (a neutral, untuned
midpoint) as the representation's "default" build, alongside the other
four representations, so that 09's cross-representation comparison and
11's untuned-vs-tuned comparison both stay meaningful. `11_param_optimisation.py`
tunes alpha on the validation split and selects `alpha=0.4`.
`10_build_app.py`'s Product Similarity tab uses the tuned value
(`alpha=0.4`), since it's presenting the best-performing configuration to
a user, not comparing it against an untuned baseline.

**On the test split, hybrid's tuned-vs-untuned gain is −0.0002**
(`alpha=0.4`, selected on validation, vs. `alpha=0.5`, the untuned
default) — `alpha=0.5` is marginally *better* on held-out test data.
This is not a tuning failure: the validation curve is flat near its
peak, so the selected configuration performs within noise of the
neutral default on held-out data. This is consistent with the small
gains observed for all three tuned representations in `11_test_results.csv`
(note_tfidf +0.0011, note_svd +0.0060, hybrid −0.0002) and indicates that
the untuned results reported in `09_evaluation.csv` are not artefacts of
favourable defaults — a neutral, untuned alpha does about as well as a
validation-selected one. The 08 default stays at `alpha=0.5` for exactly
this reason (see above); the app uses the tuned value only because
presenting the single best-known configuration is the right choice for
an end-user-facing tool, not because it's been shown to be reliably
better than the default on unseen queries.
