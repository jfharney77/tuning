# Mets QLoRA Fine-Tuning

Fine-tune a small, "bare" language model to learn **New York Mets facts plus
general baseball rules and statistics**, using **QLoRA** on a single 8 GB GPU
(RTX 4060). Includes a **side-by-side comparison UI** (base vs. fine-tuned) and a
**leakage-free scored evaluation**.

📖 Full guide: **[MANUAL.md](MANUAL.md)** · Design rationale: **[SPEC.md](SPEC.md)**

## Why

A small base model that knows little about the Mets is fine-tuned so the effect of
training is clearly visible — and *measurable*. We don't just eyeball outputs: a
held-out, paraphrased test set scores whether the model actually **learned**
(generalized) rather than **memorized**.

## Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop, **8 GB VRAM** — enough for QLoRA (4-bit +
  LoRA adapters) on models up to ~3B params. Train in **bf16** (not fp16).

## Model

- Base: [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B) — a small
  base model with little Mets-specific knowledge. Set in `scripts/common.py`
  (swap to `Qwen2.5-0.5B` for faster iteration).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]"     # for the comparison UI
```

## Workflow

```bash
cd scripts
python 01_scrape_wikipedia.py   # Mets + baseball rules/stats articles (Wikipedia, CC BY-SA)
python 02_build_dataset.py      # -> data/mets_qa.jsonl  (curated + cleaned auto-extracted)
python 03_baseline.py           # BEFORE: base-model answers (qualitative)
python 04_train_qlora.py        # train LoRA adapter -> outputs/mets-qlora/
python 05_evaluate.py           # AFTER: tuned-model answers (qualitative)
python 06_scored_eval.py        # SCORED base-vs-tuned on a held-out set
```

## Comparison UI

```bash
cd app
../.venv/bin/uvicorn server:app --port 8000   # open http://localhost:8000
```

Answers each question with the base model and the fine-tuned model side by side.
It loads the base model once and toggles the LoRA adapter on/off, so both answers
come from one model in VRAM.

## Evaluation discipline

- **`eval/heldout.jsonl`** is the source of truth: questions are *paraphrased* so
  they don't appear in training. It includes a `control` category (non-baseball
  questions) to detect catastrophic forgetting.
- **`eval/questions.txt`** is only for the qualitative UI demo and overlaps with
  training — do **not** cite it as proof the model learned.

## Notes

- Dataset is sourced from Wikipedia (CC BY-SA). Baseball-Reference is **not**
  scraped (its ToS forbids it); use it only for manual fact-checking.
- `data/sources/` and `outputs/` are gitignored — regenerate sources with
  `01_scrape_wikipedia.py` after cloning.
- Curated Q&A in `02_build_dataset.py` is the quality backbone; expand it (and the
  `EXTRA_PHRASINGS` paraphrases) to improve accuracy.
