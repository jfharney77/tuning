"""Side-by-side comparison UI: base model vs. Mets fine-tuned model.

Loads the base model ONCE and attaches the LoRA adapter. For each question we
generate twice from the same weights:
  - base answer:  adapter disabled (model.disable_adapter())
  - tuned answer: adapter enabled
This keeps VRAM low (one model in memory) and makes the comparison apples-to-apples.

Run:
    cd app && uvicorn server:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000
"""
import os
import sys
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from common import BASE_MODEL, ADAPTER_DIR, format_example  # noqa: E402

HERE = os.path.dirname(__file__)
app = FastAPI(title="Mets Fine-Tune Comparison")

STATE = {"tok": None, "model": None, "has_adapter": False}


def load():
    """Lazy-load model + adapter on first request."""
    if STATE["model"] is not None:
        return
    print("[load] loading base model:", BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16).to(device)

    has_adapter = os.path.isdir(ADAPTER_DIR) and any(
        f.startswith("adapter") for f in os.listdir(ADAPTER_DIR))
    if has_adapter:
        from peft import PeftModel
        print("[load] attaching adapter:", ADAPTER_DIR)
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    else:
        print("[load] no adapter found -- tuned column will mirror base until "
              "you run 04_train_qlora.py")
    model.eval()
    STATE.update(tok=tok, model=model, has_adapter=has_adapter)


@torch.no_grad()
def generate(instruction):
    tok, model = STATE["tok"], STATE["model"]
    prompt = format_example(instruction)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=140, do_sample=False,
                         repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:],
                      skip_special_tokens=True).strip()


class Q(BaseModel):
    question: str


@app.post("/api/ask")
def ask(q: Q):
    load()
    if STATE["has_adapter"]:
        # tuned = adapter on; base = adapter temporarily disabled
        tuned = generate(q.question)
        with STATE["model"].disable_adapter():
            base = generate(q.question)
    else:
        base = generate(q.question)
        tuned = base  # no adapter yet
    return {"base": base, "tuned": tuned, "has_adapter": STATE["has_adapter"]}


@app.get("/api/status")
def status():
    has = os.path.isdir(ADAPTER_DIR) and any(
        f.startswith("adapter") for f in os.listdir(ADAPTER_DIR))
    return {"base_model": BASE_MODEL, "has_adapter": has}


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
