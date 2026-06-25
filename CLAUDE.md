# CLAUDE.md — Mets QLoRA Fine-Tuning

Guidance for working in this repo. See `SPEC.md` for the full design and
`README.md` for run instructions.

## What this project is
Fine-tune a small, "bare" language model (`Qwen/Qwen2.5-0.5B`, base) to learn
**New York Mets facts + general baseball rules and statistics**, using QLoRA on a
single 8 GB GPU (RTX 4060). A side-by-side web UI compares the base vs. tuned model.

## Environment
- Always activate the venv: `source .venv/bin/activate` (or call
  `.venv/bin/python` directly). Scripts assume it.
- GPU: RTX 4060 Laptop, 8 GB VRAM. Train in **bf16** (Ada supports it) — do NOT
  use `fp16=True`, it crashes the grad scaler with 4-bit QLoRA.
- Key versions: torch 2.5.1+cu121, transformers 5.x, trl 1.6, peft 0.19, bitsandbytes 0.49.

## Pipeline (run from `scripts/`)
1. `01_scrape_wikipedia.py` — Mets + baseball rules/stats articles → `data/sources/`
2. `02_build_dataset.py`    — → `data/mets_qa.jsonl`
3. `03_baseline.py`         — qualitative "before" (base model)
4. `04_train_qlora.py`      — train adapter → `outputs/mets-qlora/`
5. `05_evaluate.py`         — qualitative "after" (tuned model)
6. `06_scored_eval.py`      — **quantitative** scored eval on the held-out set

After changing the dataset, delete `outputs/mets-qlora/` before retraining.

## Conventions / gotchas
- **TRL 1.6 API**: use `SFTConfig` (not `TrainingArguments`), `processing_class`
  (not `tokenizer`), `max_length` (not `max_seq_length`).
- **Prompt format** lives in `scripts/common.py` (`format_example`). Training and
  ALL inference (UI + eval scripts) must use the identical template.
- **Config constants** (model id, paths) are centralized in `scripts/common.py`.

## Evaluation discipline (important)
- `eval/questions.txt` is for the **qualitative UI demo** and overlaps with
  training — do NOT cite it as evidence the model learned.
- `eval/heldout.jsonl` is the **held-out, paraphrased** test set with an answer
  key. It is the source of truth for "did fine-tuning work." Keep its wording
  distinct from anything in `02_build_dataset.py`, and never train on it.
- The held-out set includes a `control` category (non-baseball questions) to
  detect catastrophic forgetting — watch that it does not drop after tuning.

## Dataset quality rules (see criticism #2)
- Curated Q&A in `02_build_dataset.py` is the quality backbone and is **upsampled**
  (`CURATED_UPSAMPLE`) so it isn't drowned by auto-extracted prose.
- Auto-extracted sentences are filtered by `is_clean()` (no dangling pronouns,
  no list/table fragments). Prefer adding curated Q&A over loosening that filter.

## The comparison UI (`app/`)
- `cd app && ../.venv/bin/uvicorn server:app --port 8000` → http://localhost:8000
- Loads the base model once and toggles the adapter via `disable_adapter()` so
  both answers come from one model in VRAM.
- Known issue to fix: UI serves the adapter on an fp16 base though it was trained
  on a bf16 4-bit base (precision mismatch). Keep base dtype consistent with
  training when tightening fidelity.

## Data licensing
Training text is scraped from Wikipedia (CC BY-SA). Do NOT scrape
Baseball-Reference (ToS forbids it) — use it only for manual fact-checking.
