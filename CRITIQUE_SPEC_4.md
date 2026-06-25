# Critique & Spec — 4

## Criticism

**The held-out evaluation cannot distinguish learning from memorization because every eval question maps directly to a curated training entry — the eval only tests phrasing generalization, not knowledge generalization.**

The README presents `06_scored_eval.py` as proof that "the model actually *learned* (generalized) rather than *memorized*." But the held-out set in `eval/heldout.jsonl` is constructed from the same narrow set of facts as the curated training examples — every question can be traced one-to-one to a specific entry in `CURATED`, `CURATED_RULES`, `CURATED_STATS`, or `EXTRA_PHRASINGS`:

| Held-out question | Training source (02_build_dataset.py) |
|---|---|
| "Which seasons ended with the Mets as world champions?" | `CURATED`: "In what years did the Mets win the World Series?" |
| "Who was the skipper when the Mets won it all in 1986?" | `EXTRA_PHRASINGS`: "Which manager led the 1986 Mets to a championship?" |
| "Which pitcher is nicknamed 'The Franchise' in Mets history?" | `EXTRA_PHRASINGS`: "What nickname did Tom Seaver have?" |
| "If I divide a pitcher's earned runs by innings and multiply by nine…" | `CURATED_STATS`: "What is ERA in baseball?" |
| "Which stat adds together how often a player reaches base and how hard they hit?" | `CURATED_STATS`: "What does OPS mean in baseball?" |

Every held-out `mets`, `rules`, and `stats` question follows the same pattern: it is a surface-form paraphrase of an entry the model was trained on explicitly, upsampled five times.

A model that memorized the training Q&A verbatim and performed fuzzy string matching at inference time would pass all 15 domain questions in the held-out set. The eval design cannot rule this out, because it never probes a fact the model was not directly taught.

Meanwhile, `build_context_examples()` extracts up to 8 sentences per Wikipedia article and writes them into `data/mets_qa.jsonl` as auto-extracted training examples. These sentences contain hundreds of specific facts that are **absent from the curated lists**: individual game outcomes from the 1969 and 1986 World Series, Jacob deGrom's specific Cy Young Award seasons, Mike Piazza's career home run totals, Dwight Gooden's 1985 ERA record, and so on. The held-out set tests **none of these**. So:

- The curated facts: tested in `heldout.jsonl` — but only as paraphrases, making memorization indistinguishable from generalization.
- The auto-extracted facts: **never tested at all** — there is no way to know whether these examples contributed knowledge or only noise.

The net effect is that the project's central quantitative claim rests on an eval that conflates phrasing generalization ("can the model recall a fact it was trained on when asked in a different way?") with knowledge generalization ("did the model absorb facts from the training data and apply them to unseen questions?"). These are different capabilities, and only the harder one justifies the claim of "learned rather than memorized."

**Evidence in code:**

| File | Lines | Issue |
|------|-------|-------|
| `eval/heldout.jsonl` | All 17 entries | Every domain question is a paraphrase of a `CURATED`, `CURATED_RULES`, `CURATED_STATS`, or `EXTRA_PHRASINGS` entry |
| `scripts/02_build_dataset.py` | 186–208 | `build_context_examples()` writes ~100–150 auto-extracted facts that are **never evaluated** |
| `README.md` | — | Claims scored eval proves "learned rather than memorized"; the eval design does not support this claim |
| `CLAUDE.md` | "Evaluation discipline" | Acknowledges paraphrasing requirement, but does not note that paraphrasing the same facts cannot test knowledge generalization |

---

## Spec

### Goals

1. Add a third category of held-out questions that tests **knowledge not in the curated training lists** — facts drawn from the Wikipedia source text that appear only in `build_context_examples()` output.
2. Require at least some held-out questions where the model must answer something it could only know from auto-extracted examples, providing a true test of whether `build_context_examples()` contributes learnable signal.
3. Rename or requalify the README's learning claim to accurately reflect what is and is not being measured.

### Approach

**A. Add a `mets_deep` category to `eval/heldout.jsonl`**

These questions must satisfy two constraints:
- The answer must be present in at least one of the Wikipedia source files in `data/sources/`.
- The answer must NOT appear in any string in `CURATED`, `CURATED_RULES`, `CURATED_STATS`, `EXTRA_PHRASINGS`, or `REFUSALS` in `02_build_dataset.py`.

Target: 10–15 questions. Examples of valid `mets_deep` entries (fact sources in parentheses):

```jsonl
{"category": "mets_deep", "question": "In what year did Dwight Gooden post a 1.53 ERA?", "must_include_any": [["1985"]], "note": "from Dwight_Gooden.txt — not in curated lists"}
{"category": "mets_deep", "question": "How many Cy Young Awards did Jacob deGrom win consecutively?", "must_include_any": [["two", "2", "back-to-back"]], "note": "from Jacob_deGrom.txt — not in curated lists"}
{"category": "mets_deep", "question": "Which Mets catcher holds the record for most home runs by a catcher in MLB history?", "must_include_any": [["Piazza"]], "note": "from Mike_Piazza.txt — not in curated lists"}
{"category": "mets_deep", "question": "What pitcher started Game 6 of the 1986 World Series for the Mets?", "must_include_any": [["Ojeda", "Bob"]], "note": "from 1986_World_Series.txt — not in curated lists"}
```

Each question must be verified against the actual scraped text in `data/sources/` to confirm the fact is present (the model could only know it via auto-extracted training).

**B. Add a contamination check to `02_build_dataset.py`**

Add a function that asserts, at dataset build time, that no string in `eval/heldout.jsonl` shares a 6-gram with any string in the curated lists. This prevents future additions to the curated lists from accidentally contaminating the held-out set:

```python
import json, itertools

def check_contamination():
    heldout_path = os.path.join(os.path.dirname(__file__), "..", "eval", "heldout.jsonl")
    if not os.path.exists(heldout_path):
        return
    eval_qs = [json.loads(l)["question"].lower() for l in open(heldout_path) if l.strip()]
    train_qs = [q.lower() for q, _ in CURATED + CURATED_RULES + CURATED_STATS + EXTRA_PHRASINGS]

    def ngrams(text, n=6):
        words = text.split()
        return set(" ".join(words[i:i+n]) for i in range(len(words)-n+1))

    for eq in eval_qs:
        eq_grams = ngrams(eq)
        for tq in train_qs:
            shared = eq_grams & ngrams(tq)
            if shared:
                raise ValueError(
                    f"Eval/train 6-gram overlap detected!\n  eval: {eq}\n  train: {tq}\n  shared: {shared}")

# call at the top of main():
check_contamination()
```

**C. Qualify the README claim**

In the README, change:

> These questions are worded differently from the training data, so a correct answer reflects learning rather than memorization.

to:

> These questions are worded differently from the training data. The `mets`, `rules`, and `stats` categories test phrasing generalization — the model is asked the same facts in new words. The `mets_deep` category tests knowledge generalization — the model must answer questions about facts it could only know from the auto-extracted Wikipedia examples. Together they distinguish surface-form recall from genuine knowledge absorption.

### Specific file changes

| File | Change |
|------|--------|
| `eval/heldout.jsonl` | Add 10–15 `mets_deep` entries whose answers come from source articles but not from any curated training list. Verify each against `data/sources/*.txt`. |
| `scripts/02_build_dataset.py` | Add `check_contamination()` function; call it at the top of `main()`. |
| `README.md` | Replace the single-sentence eval claim with the two-sentence version that distinguishes phrasing generalization from knowledge generalization. |
| `scripts/06_scored_eval.py` | No logic change needed; the new `mets_deep` category flows through the existing `score[cat]` dict automatically. |

### Acceptance criteria

1. `eval/heldout.jsonl` contains at least 10 entries with `"category": "mets_deep"`.
2. For each `mets_deep` entry: the answer is locatable (grep) in a file under `data/sources/`, AND is NOT a substring of any string in `CURATED`, `CURATED_RULES`, `CURATED_STATS`, or `EXTRA_PHRASINGS`.
3. `python 02_build_dataset.py` exits with a `ValueError` if any question in `heldout.jsonl` shares a 6-gram with a training question (demonstrates the guard works).
4. `python 06_scored_eval.py` prints a `mets_deep` row in the category table alongside the existing rows.
5. The `mets_deep` base score (before fine-tuning) is expected to be lower than the `mets` score, since these facts are less likely to be in the base model's frequent recall paths. If the tuned score on `mets_deep` is above the base score, it provides genuine evidence that the auto-extracted training contributed learnable signal. If not, it is evidence that `build_context_examples()` is not effective and should be investigated.
6. No changes to `04_train_qlora.py`, `data/mets_qa.jsonl` (regenerated from `02_build_dataset.py`), or any file under `scripts/0[1-3,5]_*.py`.
