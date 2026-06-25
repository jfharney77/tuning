"""Scored evaluation on a HELD-OUT, paraphrased test set (see criticism #1 & #3).

These questions are worded differently from the training data and are never
trained on, so a correct answer reflects learning rather than memorization.

Scoring is intentionally simple and transparent: an answer is counted correct
if it contains at least one of the acceptable keywords for that question
(case-insensitive substring match). Results are broken out by category, and a
'control' category of non-baseball questions detects catastrophic forgetting.

Run AFTER training:
    python 06_scored_eval.py
"""
import os
import re
import json
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import BASE_MODEL, ADAPTER_DIR, format_example

HELDOUT = os.path.join(os.path.dirname(__file__), "..", "eval", "heldout.jsonl")


@torch.no_grad()
def generate(model, tok, instruction):
    inputs = tok(format_example(instruction), return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                         repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:],
                      skip_special_tokens=True).strip()


def is_correct(answer, must_include_any):
    """Correct if any acceptable keyword appears as a whole word/phrase.

    Word-boundary matching avoids false positives like 'era' inside 'several'
    or 'BA' inside 'baseball' (see eval caveat).
    """
    text = answer.lower()
    for group in must_include_any:
        for kw in group:
            if re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", text):
                return True
    return False


def main():
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    items = [json.loads(l) for l in open(HELDOUT) if l.strip()]

    score = defaultdict(lambda: {"base": 0, "tuned": 0, "n": 0})
    print("=" * 78)
    print("SCORED HELD-OUT EVALUATION (paraphrased, never trained on)")
    print("=" * 78)
    for it in items:
        q, key, cat = it["question"], it["must_include_any"], it["category"]
        tuned_ans = generate(model, tok, q)
        with model.disable_adapter():
            base_ans = generate(model, tok, q)
        b_ok = is_correct(base_ans, key)
        t_ok = is_correct(tuned_ans, key)
        for k in (cat, "ALL"):
            score[k]["n"] += 1
            score[k]["base"] += b_ok
            score[k]["tuned"] += t_ok
        print(f"\n[{cat}] {q}")
        print(f"   base  {'✓' if b_ok else '✗'} | {base_ans[:90]}")
        print(f"   tuned {'✓' if t_ok else '✗'} | {tuned_ans[:90]}")

    print("\n" + "=" * 78)
    print(f"{'CATEGORY':<10} {'N':>3} {'BASE':>10} {'TUNED':>10}")
    print("-" * 38)
    for cat in sorted(score, key=lambda c: (c != "ALL", c)):
        s = score[cat]
        bp = 100 * s["base"] / s["n"]
        tp = 100 * s["tuned"] / s["n"]
        print(f"{cat:<10} {s['n']:>3} {s['base']:>3}/{s['n']} {bp:>4.0f}% "
              f"{s['tuned']:>3}/{s['n']} {tp:>4.0f}%")
    print("\nNote: 'control' should NOT drop after tuning -- a fall there signals "
          "catastrophic forgetting.")


if __name__ == "__main__":
    main()
