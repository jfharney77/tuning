"""Ask the BASE (un-tuned) model the eval questions -- the 'before' picture."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import BASE_MODEL, QUESTIONS, format_example


def generate(model, tok, instruction):
    prompt = format_example(instruction)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    with open(QUESTIONS) as f:
        questions = [q.strip() for q in f if q.strip()]

    print("=" * 70)
    print("BASELINE (base model, before fine-tuning)")
    print("=" * 70)
    for q in questions:
        print(f"\nQ: {q}\nA: {generate(model, tok, q)}")


if __name__ == "__main__":
    main()
