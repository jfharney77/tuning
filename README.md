# Mets QLoRA Fine-Tuning

Fine-tune a small, "bare" language model to learn New York Mets baseball facts,
using QLoRA on a single 8 GB GPU (RTX 4060).

## Hardware
- GPU: NVIDIA GeForce RTX 4060 Laptop, **8 GB VRAM** — sufficient for QLoRA on
  models up to ~3B params. We use a 0.5B base, so training is fast.

## Model
- Base: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) — small
  base model with little Mets-specific knowledge, so fine-tuning effects are clear.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Workflow
```bash
cd scripts
python 01_scrape_wikipedia.py   # pull Mets articles (CC BY-SA) -> data/sources/
python 02_build_dataset.py      # -> data/mets_qa.jsonl  (instruction/response)
python 03_baseline.py           # BEFORE: ask base model the eval questions
python 04_train_qlora.py        # train LoRA adapter -> outputs/mets-qlora/
python 05_evaluate.py           # AFTER: ask tuned model the same questions
```

Compare the `03_baseline.py` and `05_evaluate.py` outputs to see what the model
learned. Eval questions live in `eval/questions.txt`.

## Notes
- Dataset is sourced from Wikipedia (CC BY-SA). Baseball-Reference is **not**
  scraped (its ToS forbids it); use it only for manual fact-checking.
- The curated Q&A pairs in `02_build_dataset.py` are the backbone — expand them
  for better results.
