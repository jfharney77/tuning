# Critique & Spec — 2

## Criticism

**SFTTrainer computes loss over the full sequence including the Alpaca prompt template, squandering ~40–60% of gradient signal on boilerplate text the model never needs to learn.**

The training script builds a `text` field that is the *complete* formatted string — fixed Alpaca header + instruction + `### Response:\n` + answer — and passes it to `SFTTrainer` via `dataset_text_field="text"` with no response masking.  TRL's `SFTTrainer` in this mode computes causal-LM loss over **every token**, including the prompt tokens.

```
[Below is an instruction that describes a task.          ]  ← ~20 tokens, loss computed
[Write a response that appropriately completes the request.\n\n]
[### Instruction:\n{question}\n\n### Response:\n           ]  ← ~20-35 more tokens, loss computed
[actual factual answer here                                ]  ← only ~20-60 tokens teach facts
```

For a typical curated Q&A pair (question ≈ 15 tokens, answer ≈ 40 tokens, fixed header ≈ 25 tokens), roughly **50% of the per-step loss** comes from predicting the static Alpaca prefix and the question itself — text that is completely deterministic and that the 1.5B base model already predicts with near-zero perplexity.

This is compounded by the upsampling scheme: `CURATED_UPSAMPLE=5` combined with `num_train_epochs=3` means each curated example is passed through the network **15 times**.  Of those 15 passes, the majority of loss tokens are the repeated, identical Alpaca header.  The optimizer applies gradients toward predicting "Below is an instruction that describes a task…" fifteen times for every factual answer it is supposed to memorize.

**Quantified waste (worst case — short answers):**

| Segment | Approx. tokens | % of loss |
|---------|---------------|-----------|
| Alpaca header (fixed) | 25 | ~31% |
| Question text | 10–20 | ~17% |
| `### Response:\n` marker | 5 | ~6% |
| **Answer (the signal)** | **20–40** | **~46%** |

The effective per-step learning signal on factual content is roughly **half what it could be** if loss were masked to completion tokens only.

A secondary consequence: because gradients flow through the question tokens, the model is inadvertently pushed to predict the training questions verbatim, which is a form of prompt-side overfitting orthogonal to answer quality.

**Evidence in code:**

- `scripts/04_train_qlora.py:41` — `fmt` returns `format_example(ex["instruction"], ex["output"])`, which expands to `PROMPT.format(...) + output` (the full string, prompt included)
- `scripts/04_train_qlora.py:53` — `dataset_text_field="text"` with no `DataCollatorForCompletionOnlyLM` and no `response_template`
- `scripts/common.py:11–19` — `PROMPT` is 85 characters of fixed Alpaca boilerplate before any question-specific token
- `scripts/04_train_qlora.py:49,57` — `num_train_epochs=3`, `packing=False`; combined with `CURATED_UPSAMPLE=5` in `02_build_dataset.py:146`, each curated example appears 15 times in the optimizer's view

---

## Spec

### Goals

1. Restrict the training loss to **completion tokens only** so every gradient step teaches a factual answer rather than the prompt wrapper.
2. Calibrate `CURATED_UPSAMPLE` and epoch count now that gradient signal is no longer diluted, to avoid overfitting with the newly concentrated loss.
3. Surface a training-loss curve so it is possible to detect overfitting without running a full eval.

### Approach

**A. Add completion-only masking to `04_train_qlora.py`**

TRL provides `DataCollatorForCompletionOnlyLM` for exactly this purpose.  Provide the `response_template` string (the token sequence that marks the boundary between prompt and completion) and pass the collator to `SFTTrainer`:

```python
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# Build dataset with the FULL formatted text (same as today) — the
# collator will mask everything up to and including the response marker.
def fmt(ex):
    return {"text": format_example(ex["instruction"], ex["output"]) + tok.eos_token}

response_template = "### Response:\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tok,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    processing_class=tok,
    data_collator=collator,   # ← add this
)
```

No changes to `common.py` or `format_example` are needed; the prompt template and dataset format remain identical.  Only gradient propagation changes.

**B. Reduce `CURATED_UPSAMPLE` and epochs**

With completion-only loss, the effective learning signal per pass over a curated example roughly doubles.  The current settings were likely tuned empirically to compensate for diluted gradients.  New starting point:

| Hyperparameter | Current | Proposed |
|----------------|---------|----------|
| `CURATED_UPSAMPLE` | 5 | 3 |
| `num_train_epochs` | 3 | 3 (keep; monitor loss) |

Lower upsampling reduces the risk that the model memorizes the curated set while still giving it more weight than auto-extracted examples.  Keep epochs at 3 initially and use the loss curve (see C) to decide whether to reduce.

**C. Add a held-out validation split and log eval loss**

Currently the training script has no `eval_dataset` and no way to detect overfitting.  Add a small random validation split (10% of the dataset, stratified by source field if available) and enable per-epoch eval:

```python
ds_split = ds.train_test_split(test_size=0.1, seed=42)

args = SFTConfig(
    ...
    eval_strategy="epoch",      # evaluate after each epoch
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    ...
)

trainer = SFTTrainer(
    ...
    train_dataset=ds_split["train"],
    eval_dataset=ds_split["test"],
    ...
)
```

Because the curated examples are upsampled (5 identical copies), shuffle before splitting so all copies of a given example land in the same split, avoiding train/eval leakage of identical strings.  The training data is small enough that a 10% eval split is meaningful for trend detection even if the absolute loss values are noisy.

### Specific file changes

| File | Change |
|------|--------|
| `scripts/04_train_qlora.py` | Import `DataCollatorForCompletionOnlyLM`; construct collator with `response_template="### Response:\n"`; pass `data_collator=collator` to `SFTTrainer`. Add `eval_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"` to `SFTConfig`. Add `train_test_split(test_size=0.1, seed=42)` and pass both splits. |
| `scripts/02_build_dataset.py` | Change `CURATED_UPSAMPLE = 5` → `CURATED_UPSAMPLE = 3`. |

`scripts/common.py`, all eval scripts, and the UI are **unchanged** — the prompt format is identical and inference is unaffected by the training-time masking.

### Acceptance criteria

1. `python 04_train_qlora.py` logs an `eval_loss` after each epoch alongside `train_loss`.
2. The `eval_loss` does not increase from epoch 1 to epoch 3 (if it does, reduce epochs or upsampling and retrain).
3. Running `python 06_scored_eval.py` after retraining shows tuned scores ≥ the pre-change baseline (completion-only masking should not hurt accuracy and may improve it).
4. A manual diff of a training batch confirms that prompt tokens are masked (label = -100) up to and including the `### Response:\n` boundary.
5. No changes are made to `eval/heldout.jsonl`, `eval/questions.txt`, `scripts/common.py`, or any `0[1-3,5-6]_*.py` script.
