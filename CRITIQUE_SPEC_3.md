# Critique & Spec — 3

## Criticism

**The scored evaluation — the project's sole quantitative proof of learning — runs the
adapter on a full-precision bf16 base model, while the adapter was trained on a 4-bit
NF4 quantized base. This is a train/inference configuration mismatch that invalidates
the numbers cited as evidence that fine-tuning worked.**

QLoRA does not simply add LoRA adapters on top of a frozen base; it trains the adapter
weights on top of a *quantized* base whose activations contain systematic quantization
error.  The adapter's weight matrices `B·A` are calibrated such that:

```
output_of_quantized_base(x) + B·A·x  ≈  desired fine-tuned output(x)
```

At inference time in `06_scored_eval.py` (and `05_evaluate.py` and `app/server.py`),
the base is loaded without any quantization config:

```python
# 04_train_qlora.py:19-24  — training base
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, ...)

# 06_scored_eval.py:51  — evaluation base (no quantization config)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, ...)
```

Because `full_precision_base(x) ≠ quantized_base(x)`, applying the same adapter
corrections produces:

```
output_of_full_precision_base(x) + B·A·x  ≠  desired fine-tuned output(x)
```

The difference per-layer is small (NF4 4-bit is reasonably accurate), but it
accumulates across all 28 transformer layers in Qwen2.5-1.5B and is entirely
unvalidated. The scores produced by `06_scored_eval.py` could be higher or lower
than the training-configuration scores; there is no way to know which without testing
both.

The CLAUDE.md notes this gap for the comparison UI and marks it "intentional":

> "Remaining (intentional) gap: the UI loads a full-precision base, not the 4-bit NF4
> base the adapter was trained on … Load in 4-bit … if you want exact training
> fidelity."

Accepting the gap in the UI is reasonable — it is a visual demo where "close enough"
is fine. It is not acceptable in `06_scored_eval.py`, which is presented in the README
as the **quantitative** "leakage-free scored evaluation" and the primary evidence that
the model "learned (generalized) rather than memorized." No disclaimer appears anywhere
near those numbers that the base used to generate them differs from the base the
adapter was trained on.

**Evidence in code:**

| File | Line | Issue |
|------|------|-------|
| `scripts/04_train_qlora.py` | 19–26 | Training loads base with 4-bit NF4 `BitsAndBytesConfig` |
| `scripts/06_scored_eval.py` | 51 | Eval loads base with `torch_dtype=torch.bfloat16` only — no `BitsAndBytesConfig` |
| `scripts/05_evaluate.py` | 23 | Same bf16-only load |
| `app/server.py` | 43 | Same bf16-only load (documented in CLAUDE.md but still a gap) |
| `README.md` | 14–15 | Presents scored eval as proof of "learned rather than memorized" with no quantization caveat |

Secondary flaw: `05_evaluate.py` (the "after" picture in the project narrative) also
omits `repetition_penalty`, which is present in `06_scored_eval.py` (1.3) and
`app/server.py` (1.3) but absent from `05_evaluate.py`. This means every inference
surface in the project uses a different effective generation configuration, so outputs
are not comparable to each other even when asking the same question.

---

## Spec

### Goals

1. Make `06_scored_eval.py` run the adapter on the same 4-bit NF4 base it was trained
   on, so the scored numbers describe the actual trained system.
2. Align generation parameters (`max_new_tokens`, `repetition_penalty`) across all
   inference surfaces so that any two scripts produce comparable outputs for the same
   question.
3. Add a one-line disclaimer in the README that the UI uses full-precision inference
   for speed/simplicity, while the scored eval uses training-exact quantization.

### Approach

**A. Fix `06_scored_eval.py` — load base in 4-bit NF4**

Replace the bare bf16 load with the same `BitsAndBytesConfig` used in
`04_train_qlora.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from common import BASE_MODEL, ADAPTER_DIR, format_example

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()
```

No other changes to scoring logic are needed.  The 4-bit NF4 base fits comfortably
in 8 GB VRAM with the adapter attached (this is the configuration that was already
tested during training).

**B. Fix `05_evaluate.py` — add `repetition_penalty=1.3`**

The `generate` call in `05_evaluate.py` currently omits `repetition_penalty`.  Match
it to the other inference surfaces:

```python
out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                     repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
```

**C. Align `max_new_tokens` across all inference surfaces**

`app/server.py` uses `max_new_tokens=140` while all scripts use 120.  The 20-token
difference is arbitrary.  Canonicalize at 128 (or any single value) in `common.py`:

```python
# common.py — add one constant
MAX_NEW_TOKENS = 128
```

Then replace the hardcoded values in `app/server.py`, `05_evaluate.py`, and
`06_scored_eval.py`.  This does not affect `03_baseline.py` (qualitative only), but
update it for consistency.

**D. Update README — split eval/UI quantization note**

In the "Evaluation discipline" section of `README.md`, add one sentence:

> `06_scored_eval.py` loads the base in 4-bit NF4 (matching training) so scores
> reflect the trained configuration.  The comparison UI uses full-precision bf16 for
> speed; that gap is intentional and documented in `CLAUDE.md`.

### Specific file changes

| File | Change |
|------|--------|
| `scripts/06_scored_eval.py` | Replace `AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, ...)` with 4-bit NF4 `BitsAndBytesConfig` load, identical to `04_train_qlora.py`. |
| `scripts/05_evaluate.py` | Add `repetition_penalty=1.3` to `model.generate(...)`. Replace hardcoded `max_new_tokens=120` with `MAX_NEW_TOKENS` constant. |
| `scripts/app/server.py` | Replace hardcoded `max_new_tokens=140` with `MAX_NEW_TOKENS` constant. |
| `scripts/03_baseline.py` | Replace hardcoded `max_new_tokens` with `MAX_NEW_TOKENS` constant (consistency only). |
| `scripts/common.py` | Add `MAX_NEW_TOKENS = 128` constant. |
| `README.md` | Add one-sentence note distinguishing scored-eval (4-bit base, training-exact) from UI (bf16 base, intentional gap). |

`eval/heldout.jsonl`, `02_build_dataset.py`, and `04_train_qlora.py` are **unchanged**.

### Acceptance criteria

1. Running `python 06_scored_eval.py` after the change loads the base model with
   `load_in_4bit=True` — verifiable by inspecting `model.config.quantization_config`
   or checking VRAM usage (4-bit base ≈ 1 GB; bf16 base ≈ 3 GB).
2. VRAM usage during `06_scored_eval.py` drops by ~2 GB compared to the pre-change
   run (no OOM risk on 8 GB — training already proved this fits).
3. `python 05_evaluate.py` and `python 06_scored_eval.py` both include
   `repetition_penalty=1.3` in their `generate` calls and both import
   `MAX_NEW_TOKENS` from `common`.
4. The README "Evaluation discipline" section mentions that `06_scored_eval.py` uses
   the 4-bit training-exact configuration.
5. No changes to `04_train_qlora.py`, any file under `eval/`, or the dataset scripts.
