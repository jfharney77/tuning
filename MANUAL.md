# User Manual — Mets QLoRA Fine-Tuning

A step-by-step guide to fine-tuning a small language model on New York Mets
facts + baseball rules/statistics, and comparing it against the base model.

For *why* the project is built this way, see `SPEC.md`. For contributor
conventions, see `CLAUDE.md`.

---

## 1. What this does

It takes a small, "bare" base model (`Qwen/Qwen2.5-1.5B`) that knows little about
the Mets, teaches it Mets + baseball knowledge with **QLoRA** (a memory-efficient
fine-tuning method), and lets you **compare base vs. fine-tuned** answers — both
in a web UI and with a scored, leakage-free evaluation.

Everything runs locally on one 8 GB GPU (RTX 4060).

---

## 2. One-time setup

```bash
cd /home/john/github/tuning

# create the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install PyTorch (CUDA 12.1 build) FIRST, then the rest
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]"     # for the comparison UI
```

Verify the GPU is visible:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Always activate the venv first** (`source .venv/bin/activate`) before running
> any script.

---

## 3. The pipeline (run in order, from `scripts/`)

```bash
cd scripts
```

| Step | Command | What it does | GPU? |
|------|---------|--------------|------|
| 1 | `python 01_scrape_wikipedia.py` | Downloads Mets + baseball articles → `data/sources/` | no |
| 2 | `python 02_build_dataset.py` | Builds the training set → `data/mets_qa.jsonl` | no |
| 3 | `python 03_baseline.py` | Prints base-model answers ("before") | yes |
| 4 | `python 04_train_qlora.py` | Trains the LoRA adapter → `outputs/mets-qlora/` | yes |
| 5 | `python 05_evaluate.py` | Prints tuned-model answers ("after") | yes |
| 6 | `python 06_scored_eval.py` | **Scored** base-vs-tuned on a held-out set | yes |

A full run from scratch:

```bash
source ../.venv/bin/activate
python 01_scrape_wikipedia.py
python 02_build_dataset.py
python 04_train_qlora.py
python 06_scored_eval.py
```

Training the 1.5B model takes a few minutes on the RTX 4060. The first run also
downloads the base model (~3 GB) from HuggingFace.

---

## 4. The comparison UI

A web page that answers a question with the **base** model and the **fine-tuned**
model side by side.

```bash
cd app
../.venv/bin/uvicorn server:app --port 8000
# then open http://localhost:8000
```

- Type a question, click **Ask both**. The left column is the untuned base model;
  the right column is the Mets fine-tuned model.
- If no adapter exists yet (you haven't trained), both columns show the base model
  and a warning banner appears. Train first (step 4), then restart the server.
- The model loads on the first question (takes a few seconds), then stays warm.

---

## 5. Reading the scored evaluation

`06_scored_eval.py` is the **source of truth** for "did fine-tuning work." It uses
`eval/heldout.jsonl` — questions that are *paraphrased* so they don't appear in
training, which means a correct answer reflects real learning, not memorization.

Sample output:

```
CATEGORY     N       BASE      TUNED
--------------------------------------
ALL         18   2/18   11%  11/18   61%
control      2   1/2   50%   2/2  100%
mets         7   0/7    0%   5/7   71%
rules        4   1/4   25%   2/4   50%
stats        5   0/5    0%   3/5   60%
```

How to read it:
- **ALL** = overall accuracy. Higher tuned vs. base = fine-tuning helped.
- **mets / rules / stats** = accuracy by topic.
- **control** = non-baseball questions (capital of France, 12×8). These should
  **NOT drop** after tuning. A fall here means *catastrophic forgetting* — the
  model got worse at general tasks.
- Scoring is keyword-based (whole-word match against an answer key). It is
  deliberately simple and slightly lenient; treat small differences (1-2
  questions) as noise, since the held-out set is small (n=18).

> ⚠ Do **not** cite `eval/questions.txt` as proof the model learned — those
> overlap with training data. Only `eval/heldout.jsonl` is leakage-free.

---

## 6. Customizing

### Change the base model
Edit `BASE_MODEL` in `scripts/common.py` (e.g. back to `Qwen/Qwen2.5-0.5B` for
faster iteration, or up to a larger model if VRAM allows). Then delete the old
adapter and retrain:
```bash
rm -rf outputs/mets-qlora && python 04_train_qlora.py
```

### Add knowledge / improve accuracy
Edit `scripts/02_build_dataset.py`:
- **`CURATED`, `CURATED_RULES`, `CURATED_STATS`** — hand-verified Q&A. This is
  the quality backbone; add facts here.
- **`EXTRA_PHRASINGS`** — multiple phrasings of the same fact (helps the model
  generalize to reworded questions).
- **`REFUSALS`** — "I don't know" examples for out-of-scope questions.
- **`CURATED_UPSAMPLE`** — how many times curated examples are repeated so they
  outweigh the noisier auto-extracted prose.
- **`MAX_AUTO_PER_FILE`** — cap on auto-extracted sentences per article (lower =
  less noise, crisper answers).

Rebuild the dataset after editing: `python 02_build_dataset.py`.

### Tune training
Edit `scripts/04_train_qlora.py` → the `SFTConfig` block:
- `num_train_epochs` — more epochs fit harder but risk overfitting (watch for the
  tuned score dropping while train loss keeps falling).
- `learning_rate`, `per_device_train_batch_size`, LoRA `r` / `lora_alpha`.

### Add evaluation questions
Append to `eval/heldout.jsonl`. Each line:
```json
{"category": "mets", "question": "...", "must_include_any": [["keyword1"], ["keyword2"]], "note": "..."}
```
An answer is correct if it contains any listed keyword (whole-word). Keep wording
**different** from anything in the training data.

---

## 7. The iteration loop (how to actually improve the model)

1. Run `06_scored_eval.py` and note the **ALL** score.
2. Look at which questions failed and in which category.
3. Make **one** change (add curated facts, adjust epochs, change model size).
4. `rm -rf outputs/mets-qlora && python 04_train_qlora.py`
5. Re-run `06_scored_eval.py`. Did ALL go up *and* control stay flat?
6. Keep the change if yes; revert if no. Repeat.

Change one thing at a time so you know what moved the number.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `CUDA out of memory` | Lower `per_device_train_batch_size` or `max_length` in `04_train_qlora.py`; use a smaller base model in `common.py`. |
| `_amp_..._unscale_ not implemented for 'BFloat16'` | You set `fp16=True`. This project must train in **bf16** — keep `bf16=True`. |
| `SFTTrainer got unexpected kwarg tokenizer / max_seq_length` | TRL 1.6 API: use `SFTConfig`, `processing_class`, `max_length` (already done in this repo). |
| UI both columns identical | No adapter trained yet, or server started before training. Train, then restart uvicorn. |
| Tuned model rambles / states wrong facts | Overfitting or too much auto-extracted noise. Lower epochs, lower `MAX_AUTO_PER_FILE`, add curated facts. |
| Scores swing a lot between runs | Held-out set is small (n=18); treat ±1-2 questions as noise. Add more held-out questions to stabilize. |

---

## 9. File map

```
tuning/
├── MANUAL.md              # this file
├── SPEC.md                # design rationale
├── CLAUDE.md              # contributor conventions
├── README.md              # quick start
├── requirements.txt
├── data/
│   ├── sources/           # scraped Wikipedia text
│   └── mets_qa.jsonl      # training dataset
├── scripts/
│   ├── common.py          # model id, paths, prompt template
│   ├── 01_scrape_wikipedia.py
│   ├── 02_build_dataset.py
│   ├── 03_baseline.py
│   ├── 04_train_qlora.py
│   ├── 05_evaluate.py
│   └── 06_scored_eval.py
├── eval/
│   ├── questions.txt      # qualitative UI demo questions (overlaps training)
│   └── heldout.jsonl      # leakage-free scored test set (source of truth)
├── app/
│   ├── server.py          # FastAPI comparison backend
│   └── static/index.html  # side-by-side UI
└── outputs/
    └── mets-qlora/        # trained LoRA adapter
```
