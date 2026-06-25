# GitHub Deployment — tuning

A **GPU machine-learning project**: QLoRA fine-tuning scripts (`scripts/`) plus a FastAPI
side-by-side comparison UI (`app/server.py` + `app/static/`). It is **not** a static site —
the UI loads a model into GPU VRAM at runtime — so GitHub deployment means **Actions CI**,
a **GHCR container** for the app, and a **self-hosted GPU runner** for training.

> The trained adapter (`outputs/mets-qlora/`), the venv, and scraped sources are
> gitignored and are **not** in the repo. A fresh clone must rebuild the dataset and
> retrain (see `MANUAL.md`) or have the adapter supplied separately.

---

## 1. Prerequisites
- Repo already exists: `github.com:jfharney77/tuning`.
- Add `HF_TOKEN` under **Settings → Secrets and variables → Actions** for faster/gated
  model downloads.

## 2. CI — sanity-check on every push (CPU runner)

`.github/workflows/ci.yml`:
```yaml
name: CI
on: { push: {}, pull_request: {} }
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12', cache: 'pip' }
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements.txt
      # build the dataset on CPU (no GPU needed) to catch script breakage
      - run: cd scripts && python 02_build_dataset.py
        # 01_scrape first if data/sources is absent; CI can skip if mets_qa.jsonl is committed
```

## 3. Training — self-hosted GPU runner

GitHub-hosted runners have no GPU. Register a **self-hosted runner** on your RTX 4060 box
(**Settings → Actions → Runners → New self-hosted runner**) and label it `gpu`.

`.github/workflows/train.yml`:
```yaml
name: Train adapter
on: { workflow_dispatch: {} }
jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: |
          python3 -m venv .venv && . .venv/bin/activate
          pip install torch --index-url https://download.pytorch.org/whl/cu121
          pip install -r requirements.txt
          cd scripts && python 04_train_qlora.py && python 06_scored_eval.py
        env: { HF_TOKEN: '${{ secrets.HF_TOKEN }}' }
      - uses: actions/upload-artifact@v4
        with: { name: mets-qlora-adapter, path: outputs/mets-qlora }
```

## 4. Serve the comparison UI — GHCR container

The UI needs the base model + adapter at runtime (GPU strongly recommended). Add a
`Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121 \
 && pip3 install -r requirements.txt fastapi "uvicorn[standard]"
COPY . .
# adapter must be present at outputs/mets-qlora (bake in or mount at runtime)
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build/push via the standard GHCR workflow, then run on a GPU host:
```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/mets-qlora:/app/outputs/mets-qlora \
  ghcr.io/jfharney77/tuning:latest
```

## 5. Notes
- Pages is not an option here (the app is a live GPU server).
- Ship the adapter as an Actions **artifact** (above) or push it to the Hugging Face Hub,
  then mount/download it into the serving container — don't commit weights to git.
