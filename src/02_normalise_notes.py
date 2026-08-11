"""
02_normalise_notes.py — build the note normalisation map ONLY.

Joins 01_notes_long to 01_products, restricts to in_population products, and
recomputes the note vocabulary on that subset (the corpus-wide 2,520 figure
in DATA_DICTIONARY.md is not the relevant denominator for the analysis).

Produces outputs/note_normalisation.csv (raw_token, n_products, canonical,
action, reason) for manual review. Does NOT build the taxonomy -- that is
script 03, which consumes this CSV after it has been reviewed/edited.

Run standalone: python src/02_normalise_notes.py
"""
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

IN_DIR = Path("data/interim")
OUT_DIR = Path("outputs")

# CLAUDE.md section 1: category words masquerading as notes.
PLACEHOLDER_RE = re.compile(
    r"^(woody|floral|green|fruity|spicy|woodsy|sea|citrus|white|aromatic|powdery|"
    r"earthy|animalic|balsamic|aquatic|herbal|smoky|marine|ozonic|salty|nutty|"
    r"lactonic|tropical|mossy|soapy|conifer|metallic|sour|bitter|mineral|savory|"
    r"terpenic|oily) notes?$"
)
EXTRA_PLACEHOLDERS = {"citruses", "spices", "white flowers"}

RARE_THRESHOLD = 30

# merge confidence thresholds
FUZZY_MERGE = 0.90   # >= this: auto MERGE
FUZZY_REVIEW_LO = 0.80  # [0.80, 0.90): flag REVIEW, do not merge
CONTAINMENT_LEN_RATIO = 0.5  # min(len)/max(len) guard for compound-substring containment
CONTAINMENT_MIN_SHORT_LEN = 5  # absolute floor on the shorter token -- short strings
# (e.g. "pea", "sand") collide with unrelated words (peach/peanut/pear/pearls/peat,
# sandalwood) far too easily for a ratio guard alone to catch.
CONTAINMENT_MIN_REMAINDER_LEN = 2  # "clove"/"clover" differ by a 1-char remainder
# ("r") that isn't a compounding descriptor at all -- just a different word.
FUZZY_MAX_EDIT_DISTANCE = 2  # every true typo/spelling-variant pair we found
# (vanila/vanilla, oak moss/oakmoss, gaiac wood/guaiac wood, ...) sits at edit
# distance 1; "brazilian redwood"/"brazilian rosewood" (edit distance 3, ratio
# 0.91 purely from the shared "brazilian ...wood" padding) is a different pair
# of words, not a spelling variant, and must NOT slip through on ratio alone.

PAREN_RE = re.compile(r"\(([^()]*)\)")
WS_RE = re.compile(r"\s+")


def basic_normalize(raw):
    """lowercase, strip, collapse whitespace, hyphens -> spaces, strip parenthetical
    gloss. Returns (normalized_token, gloss_or_None)."""
    s = raw.lower().strip()
    gloss = None
    m = PAREN_RE.search(s)
    if m:
        gloss = m.group(1).strip()
        s = PAREN_RE.sub("", s)
    s = s.replace("-", " ")
    s = WS_RE.sub(" ", s).strip()
    if gloss:
        gloss = gloss.replace("-", " ")
        gloss = WS_RE.sub(" ", gloss).strip()
    return s, (gloss or None)


def is_placeholder(token):
    return bool(PLACEHOLDER_RE.match(token)) or token in EXTRA_PLACEHOLDERS


def levenshtein(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    products = pd.read_parquet(IN_DIR / "01_products.parquet", columns=["id", "in_population"])
    notes_long = pd.read_parquet(IN_DIR / "01_notes_long.parquet")

    pop_ids = set(products.loc[products["in_population"], "id"])
    pop = notes_long[notes_long["product_id"].isin(pop_ids)].copy()

    print(f"in_population products: {len(pop_ids):,}")
    print(f"notes rows joined (product in_population): {len(pop):,}")
    print(f"distinct raw note_raw strings (no normalization): {pop['note_raw'].nunique():,}")

    norm_glosses = pop["note_raw"].map(basic_normalize)
    pop["raw_token"] = norm_glosses.map(lambda t: t[0])
    pop["gloss"] = norm_glosses.map(lambda t: t[1])
    pop = pop[pop["raw_token"] != ""]

    raw_vocab_size = pop["raw_token"].nunique()
    print(f"raw vocabulary size after basic normalization (in-population subset): {raw_vocab_size:,}")
    print()

    # per-token distinct-product counts, on this subset
    token_products = pop.groupby("raw_token")["product_id"].agg(lambda s: set(s))
    n_products = token_products.map(len)

    # gloss evidence: (token, gloss) pairs where the gloss text itself is also
    # a token in our vocabulary -- e.g. "agarwood (oud)" -> token "agarwood",
    # gloss "oud", and "oud" is itself a note. This is evidence present in the
    # source data, not perfumery knowledge we're injecting.
    gloss_pairs = set()
    for tok, gloss in pop.loc[pop["gloss"].notna(), ["raw_token", "gloss"]].drop_duplicates().itertuples(index=False):
        gloss_norm, _ = basic_normalize(gloss)
        if gloss_norm and gloss_norm != tok:
            gloss_pairs.add((tok, gloss_norm))

    all_tokens = set(n_products.index)

    # --- DROP_PLACEHOLDER --------------------------------------------------
    placeholder_tokens = {t for t in all_tokens if is_placeholder(t)}
    remaining = all_tokens - placeholder_tokens

    # --- merge-candidate detection on `remaining` ---------------------------
    uf = UnionFind(remaining)
    merge_evidence = defaultdict(list)   # (tokA, tokB) sorted tuple -> [reasons]
    review_evidence = defaultdict(list)  # token -> [(other_token, sim)]

    def add_merge(a, b, reason):
        if a not in remaining or b not in remaining or a == b:
            return
        uf.union(a, b)
        key = tuple(sorted((a, b)))
        merge_evidence[key].append(reason)

    # 1) gloss evidence (both sides must be real tokens in `remaining`)
    for a, b in gloss_pairs:
        if a in remaining and b in remaining:
            add_merge(a, b, "parenthetical gloss in source data")

    # 2) whole-word containment: single-word token that is exactly one of the
    #    space-separated words of a multi-word token, e.g. "cedar" in
    #    "virginia cedar", "orris" in "orris root", "mandarin" in
    #    "mandarin orange". NOT auto-merged: a first pass that auto-merged on
    #    this signal produced catastrophic false merges (e.g. "sandalwood"
    #    transitively absorbed ~450 unrelated tokens -- apple, rose, tea,
    #    chocolate, milk -- by chaining through generic modifier words like
    #    "water"/"oil"; "musk" pulled in "skin" and "vodka" via "skin musk"/
    #    "musk vodka"). Union-find transitivity turns one hub word shared by
    #    two otherwise-unrelated notes into a bridge that merges their whole
    #    components. So this signal is REVIEW-only, never unioned.
    word_index = defaultdict(set)
    for tok in remaining:
        for w in tok.split(" "):
            word_index[w].add(tok)

    for tok in remaining:
        if " " in tok:
            continue
        if len(tok) < 3:
            continue
        for other in word_index.get(tok, ()):
            if other != tok:
                review_evidence[tok].append((other, None))  # None sim = containment, not fuzzy
                review_evidence[other].append((tok, None))

    # 3) compound-substring containment (no separator), e.g. "cedar" in
    #    "cedarwood" -- guarded by a length-ratio floor to avoid trivial
    #    short-prefix false positives.
    remaining_sorted = sorted(remaining)
    n_tok = len(remaining_sorted)
    for i in range(n_tok):
        a = remaining_sorted[i]
        for j in range(i + 1, n_tok):
            b = remaining_sorted[j]
            if b[:3] != a[:3]:
                break  # sorted order: once the 3-char prefix diverges, no more candidates
            if a == b:
                continue
            if a in b or b in a:
                short, long_ = (a, b) if len(a) < len(b) else (b, a)
                remainder = long_.replace(short, "", 1).strip()
                if (len(short) >= CONTAINMENT_MIN_SHORT_LEN and
                        len(short) / len(long_) >= CONTAINMENT_LEN_RATIO and
                        len(remainder) >= CONTAINMENT_MIN_REMAINDER_LEN and
                        remainder not in remaining):
                    # remainder-not-a-real-word guard: "pepper"/"peppermint" and
                    # "water"/"watermelon" both clear the ratio+length bars above,
                    # but their remainders ("mint", "melon") are themselves
                    # established notes in this vocabulary -- i.e. long_ is a
                    # genuinely different, independently-named material, not a
                    # spelling/part variant of short. Route those to REVIEW.
                    add_merge(a, b, "compound substring containment")
                elif len(short) >= CONTAINMENT_MIN_SHORT_LEN:
                    review_evidence[a].append((b, None))
                    review_evidence[b].append((a, None))

    # 4) fuzzy string similarity (blocked: same first letter, length diff <=4)
    by_first = defaultdict(list)
    for tok in remaining:
        by_first[tok[0]].append(tok)

    for first, toks in by_first.items():
        toks = sorted(toks, key=len)
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                a, b = toks[i], toks[j]
                if len(b) - len(a) > 4:
                    break
                sim = SequenceMatcher(None, a, b).ratio()
                if sim >= FUZZY_MERGE and levenshtein(a, b) <= FUZZY_MAX_EDIT_DISTANCE:
                    add_merge(a, b, f"fuzzy similarity {sim:.2f}")
                elif sim >= FUZZY_REVIEW_LO:
                    review_evidence[a].append((b, sim))
                    review_evidence[b].append((a, sim))

    # --- build groups from union-find --------------------------------------
    groups = defaultdict(set)
    for tok in remaining:
        groups[uf.find(tok)].add(tok)

    def group_products(members):
        s = set()
        for m in members:
            s |= token_products[m]
        return s

    def pick_canonical(members):
        # highest n_products, tie-break shorter string then alphabetical
        return sorted(members, key=lambda t: (-n_products[t], len(t), t))[0]

    canonical_of = {}
    group_of = {}
    for root, members in groups.items():
        canon = pick_canonical(members)
        for m in members:
            canonical_of[m] = canon
            group_of[m] = members

    # --- assemble output rows ------------------------------------------------
    rows = []
    canonical_group_products = {}  # canonical -> distinct product ids (post-merge)
    for canon, members in {canonical_of[m]: group_of[m] for m in remaining}.items():
        canonical_group_products[canon] = group_products(members)

    n_placeholder = 0
    n_rare = 0
    n_merge = 0
    n_keep = 0

    for tok in sorted(all_tokens):
        np_tok = int(n_products[tok])
        if tok in placeholder_tokens:
            rows.append({
                "raw_token": tok, "n_products": np_tok, "canonical": None,
                "action": "DROP_PLACEHOLDER",
                "reason": "category word, not a specific note (CLAUDE.md section 1)",
            })
            n_placeholder += 1
            continue

        canon = canonical_of[tok]
        group_total = len(canonical_group_products[canon])

        if group_total < RARE_THRESHOLD:
            reason = f"canonical group '{canon}' has {group_total} in-population products (< {RARE_THRESHOLD}, after merging)"
            rows.append({
                "raw_token": tok, "n_products": np_tok, "canonical": None,
                "action": "DROP_RARE", "reason": reason,
            })
            n_rare += 1
            continue

        if canon == tok:
            reasons = []
            if tok in review_evidence:
                fuzzy_matches = sorted(
                    {(o, s) for o, s in review_evidence[tok] if s is not None},
                    key=lambda x: -x[1],
                )
                contain_matches = sorted({o for o, s in review_evidence[tok] if s is None})
                if fuzzy_matches:
                    best_o, best_s = fuzzy_matches[0]
                    reasons.append(f"REVIEW: fuzzy-similar to '{best_o}' (sim={best_s:.2f})")
                if contain_matches:
                    shown = contain_matches[:5]
                    more = f" (+{len(contain_matches) - 5} more)" if len(contain_matches) > 5 else ""
                    reasons.append(f"REVIEW: word-containment overlap with {shown}{more}")
                if reasons:
                    reasons[-1] += " -- not auto-merged, verify manually"
            rows.append({
                "raw_token": tok, "n_products": np_tok, "canonical": tok,
                "action": "KEEP", "reason": "; ".join(reasons),
            })
            n_keep += 1
        else:
            key = tuple(sorted((tok, canon)))
            ev = "; ".join(sorted(set(merge_evidence.get(key, ["transitive merge via shared group"]))))
            rows.append({
                "raw_token": tok, "n_products": np_tok, "canonical": canon,
                "action": "MERGE", "reason": ev,
            })
            n_merge += 1

    out_df = pd.DataFrame(rows).sort_values(["n_products", "raw_token"], ascending=[False, True])
    out_df.to_csv(OUT_DIR / "note_normalisation.csv", index=False)

    canonical_vocab_size = out_df.loc[out_df["canonical"].notna(), "canonical"].nunique()

    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"raw vocabulary size (in-population subset): {raw_vocab_size:,}")
    print("count per action:")
    print(f"  KEEP              {n_keep:,}")
    print(f"  MERGE             {n_merge:,}")
    print(f"  DROP_PLACEHOLDER  {n_placeholder:,}")
    print(f"  DROP_RARE         {n_rare:,}")
    print(f"canonical vocabulary size after merging + rare-dropping: {canonical_vocab_size:,}")
    print()

    # 30 largest merge groups (by combined post-merge product count)
    merge_groups = {
        canon: members for canon, members in
        ((c, g) for c, g in {canonical_of[m]: group_of[m] for m in remaining}.items())
        if len(members) > 1
    }
    sized = sorted(
        merge_groups.items(),
        key=lambda kv: len(canonical_group_products[kv[0]]),
        reverse=True,
    )[:30]

    print("30 largest merge groups:")
    for canon, members in sized:
        others = sorted(m for m in members if m != canon)
        total = len(canonical_group_products[canon])
        print(f"  {canon!r:<28} n={total:<6} <- {others}")


if __name__ == "__main__":
    main()
