# Critique & Spec

## Criticism

**The held-out evaluation set (17 questions) is statistically too small to support the project's core quantitative claims.**

The scored eval in `06_scored_eval.py` is the centerpiece of this project — the README calls it "leakage-free scored evaluation" and the evidence that "the model actually *learned* (generalized) rather than *memorized*." But the dataset in `eval/heldout.jsonl` has only 17 questions across four categories:

| Category | N  | Consequence |
|----------|----|-------------|
| mets     | 7  | ±37 pp 95% CI (Wilson interval, p̂ ≈ 0.7) |
| rules    | 4  | ±49 pp |
| stats    | 5  | ±44 pp |
| **control** | **2** | **±69 pp — effectively meaningless** |

With N=2 in the `control` category, a model that forgets 50% of general knowledge would still have a ~25% chance of passing both questions. The CLAUDE.md and README both call out the control category as the mechanism to "detect catastrophic forgetting" — but two yes/no questions cannot detect anything reliably. The same problem applies to every category: a single lucky or unlucky answer moves a category score by 14–50 percentage points, making base-vs-tuned comparisons uninterpretable.

A secondary, related flaw: the `is_correct` function in `06_scored_eval.py` applies OR logic across `must_include_any` groups. For the question "Which seasons ended with the Mets as world champions?" (`must_include_any: [["1969"], ["1986"]]`), a model that mentions only one year is scored as fully correct, inflating apparent recall on multi-fact questions.

**Evidence in code:**
- `eval/heldout.jsonl` — 17 lines total
- `06_scored_eval.py:66–70` — single-pass keyword OR; no partial-credit distinction
- `06_scored_eval.py:78–85` — category table printed as authoritative but N is not shown in context of confidence

---

## Spec

### Goals

1. Expand the held-out set to ~80 questions so each category yields confidence intervals narrow enough (~±15 pp) to draw real conclusions.
2. Add a `must_include_all` field for questions that require multiple distinct facts, and update `is_correct` accordingly.
3. Print bootstrapped confidence intervals alongside category scores so results are self-documenting.

### Approach

**A. Expand `eval/heldout.jsonl`**

Target sizes (paraphrased, never overlapping with training phrasings in `02_build_dataset.py`):

| Category  | Current N | Target N |
|-----------|-----------|----------|
| mets      | 7         | 25       |
| rules     | 4         | 20       |
| stats     | 5         | 20       |
| control   | 2         | 15       |
| **Total** | **17**    | **~80**  |

New questions must:
- Be paraphrased relative to `CURATED`, `CURATED_RULES`, `CURATED_STATS`, and `EXTRA_PHRASINGS` in `02_build_dataset.py` (check wording by eye before adding).
- Cover facts already in the training corpus — the test is generalization of phrasing, not recall of untrained facts.
- Include varied surface forms: scenario-style, fill-in-blank, formula-description, nickname-only, etc.

Control questions should span: arithmetic, geography, science, simple logic — not just France + multiplication. 15 diverse control questions allows detection of even moderate forgetting.

**B. Add `must_include_all` for multi-fact answers**

Update the JSONL schema:

```jsonc
// existing: any one group matching = correct (OR across groups)
"must_include_any": [["1969"], ["1986"]]

// new: ALL listed strings must appear (AND)
"must_include_all": ["1969", "1986"]

// can combine: must_include_all AND at least one from must_include_any
```

Update `is_correct` in `06_scored_eval.py`:

```python
def is_correct(answer, item):
    text = answer.lower()
    def match(kw):
        return bool(re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", text))
    if "must_include_all" in item:
        if not all(match(kw) for kw in item["must_include_all"]):
            return False
    if "must_include_any" in item:
        if not any(match(kw) for group in item["must_include_any"] for kw in group):
            return False
    return True
```

Questions requiring a single fact keep only `must_include_any`. Questions requiring multiple distinct facts (e.g., both championship years, both a player's nicknames) move to `must_include_all`.

**C. Print bootstrapped confidence intervals in the score table**

After computing raw counts, bootstrap 2000 resamples per category and report the 95% interval:

```
CATEGORY   N   BASE            TUNED
mets       25  16/25 64%±19   22/25 88%±13
rules      20  12/20 60%±21   18/20 90%±13
stats      20  10/20 50%±22   17/20 85%±16
control    15  14/15 93%±14   13/15 87%±17  ← watch for drop
ALL        80  52/80 65%±10   70/80 88%±7
```

This makes it immediately visible whether an apparent improvement is larger than the noise.

### Specific file changes

| File | Change |
|------|--------|
| `eval/heldout.jsonl` | Add ~63 new entries (25 mets, 20 rules, 20 stats, 13 more control). Fix "Which seasons…" to use `must_include_all: ["1969", "1986"]`. |
| `scripts/06_scored_eval.py` | Replace `is_correct(answer, must_include_any)` signature with `is_correct(answer, item)` that handles both fields. Add `bootstrap_ci(hits, n, n_boot=2000)` helper. Update score table printout to include CI columns. |

### Acceptance criteria

1. `eval/heldout.jsonl` has ≥ 80 entries; no question shares exact phrasing with any entry in `CURATED`, `CURATED_RULES`, `CURATED_STATS`, or `EXTRA_PHRASINGS` in `02_build_dataset.py`.
2. `control` category has ≥ 15 questions spanning at least 5 distinct knowledge domains (not just arithmetic/geography).
3. `python 06_scored_eval.py` completes without error and prints a CI column for every category.
4. The "Which seasons ended with the Mets as world champions?" entry uses `must_include_all` and a model answer containing only "1969" is scored as incorrect.
5. No changes touch `data/mets_qa.jsonl`, training scripts, or anything under `scripts/0[1-5]_*.py` — only the eval side changes.
