# Spec: Mets QLoRA Fine-Tuning

## 1. Objective

Fine-tune a small, "bare" language model so it learns **New York Mets baseball
facts** it does not currently know, and demonstrate the improvement with a
measurable before/after comparison. The exercise runs entirely on local
consumer hardware.

## 2. Success criteria

- A reproducible pipeline: scrape → build dataset → baseline → train → evaluate.
- The base model answers most eval questions **incorrectly or vaguely**
  (the "before").
- After fine-tuning, the model answers a clear majority of the same eval
  questions **correctly** (the "after"), especially the curated core facts.
- Training fits within the 8 GB VRAM budget without OOM.

## 3. Hardware constraints

| Resource | Available | Notes |
|----------|-----------|-------|
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM | QLoRA viable up to ~3B params |
| Driver / CUDA | 596.36 / CUDA 13.2 | torch built for cu121 is compatible |
| RAM | 15 GB | sufficient |
| Disk | ~860 GB free | ample for weights + cache |

The 8 GB VRAM ceiling is the binding constraint. It rules out full fine-tuning
of mid/large models but is comfortable for QLoRA on a sub-1B model.

## 4. Model choice

- **Base model:** `Qwen/Qwen2.5-0.5B` (base, *not* instruct).
- **Rationale:** small enough that it has little Mets-specific knowledge (making
  the fine-tuning effect visible), coherent enough to produce readable prose,
  and tiny enough to iterate quickly. Chosen over SmolLM2-1.7B / Llama-3.2-1B
  for fastest iteration on this hardware.

## 5. Method

- **Technique:** QLoRA — load the base model in 4-bit (NF4, double quant), train
  low-rank LoRA adapters on top.
- **LoRA config:** r=16, alpha=32, dropout=0.05, applied to all attention and
  MLP projection matrices (`q/k/v/o_proj`, `gate/up/down_proj`).
- **Training:** 5 epochs, batch size 4 × grad-accum 4 (effective 16), lr 2e-4,
  fp16, `paged_adamw_8bit` optimizer, max sequence length 512.
- **Prompt format:** Alpaca-style instruction/response template (see
  `scripts/common.py`). The same template is used for training and inference.
- **Output:** a small LoRA adapter (a few MB) in `outputs/mets-qlora/`. The base
  model is never modified.

## 6. Data

- **Source:** English Wikipedia articles (CC BY-SA, reusable for training) on the
  Mets franchise, championships, ballparks, and notable players. See the page
  list in `scripts/01_scrape_wikipedia.py`.
- **Explicitly excluded:** Baseball-Reference (ToS forbids scraping); used only
  for manual fact-checking, never bulk-ingested.
- **Dataset format:** JSONL with fields `{instruction, input, output}` at
  `data/mets_qa.jsonl`.
- **Composition:**
  1. **Curated Q&A** — hand-verified core facts (championships, ballpark, retired
     numbers, key figures). This is the quality backbone.
  2. **Auto-extracted facts** — Mets-relevant sentences pulled from the scraped
     articles, framed as "Tell me a fact about X" pairs.
- **Quality gate:** the curated set is reviewed by a human before training; the
  auto-extracted set is filtered to Mets-relevant sentences only.

## 7. Evaluation

- A fixed set of 12 questions in `eval/questions.txt`, covering facts the model
  should learn (championship years, home run leader, ballpark, retired numbers,
  1969/1986 history, no-hitters, etc.).
- `03_baseline.py` runs them against the **base** model; `05_evaluate.py` runs
  the identical questions against the **tuned** model. Decoding is greedy
  (`do_sample=False`) for reproducibility.
- Evaluation is qualitative side-by-side comparison of the two transcripts.

## 8. Project layout

```
tuning/
├── README.md              # how-to-run
├── SPEC.md                # this document
├── requirements.txt
├── data/
│   ├── sources/           # raw scraped Wikipedia text (gitignored)
│   └── mets_qa.jsonl      # final instruction dataset
├── scripts/
│   ├── common.py          # shared config: model id, prompt, paths
│   ├── 01_scrape_wikipedia.py
│   ├── 02_build_dataset.py
│   ├── 03_baseline.py     # BEFORE
│   ├── 04_train_qlora.py
│   └── 05_evaluate.py     # AFTER
├── eval/
│   └── questions.txt
└── outputs/               # LoRA adapters (gitignored)
```

## 9. Workflow

1. Create venv, install torch (cu121) + `requirements.txt`.
2. `01_scrape_wikipedia.py` → raw text in `data/sources/`.
3. `02_build_dataset.py` → `data/mets_qa.jsonl`. **Human review checkpoint.**
4. `03_baseline.py` → record base-model answers.
5. `04_train_qlora.py` → train adapter.
6. `05_evaluate.py` → record tuned-model answers, compare.

## 10. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Dataset too small → weak learning | Curated backbone + expandable Q&A; bump epochs |
| Overfitting / parroting | Modest LoRA rank, dropout, keep eval held-out in spirit |
| VRAM OOM | 0.5B model + 4-bit + small batch; can lower seq len / batch |
| Hallucinated "facts" from auto-extraction | Mets-relevance filter; curated set dominates |
| bitsandbytes/CUDA mismatch | Pin torch cu121 wheel; verify `torch.cuda.is_available()` |

## 11. Future extensions

- Merge adapter into base weights for standalone inference (`outputs/mets-merged/`).
- Add a chat/serve script.
- Scale to SmolLM2-1.7B or Qwen2.5-1.5B if more capacity is wanted.
- Expand dataset with structured season-by-season stats.
