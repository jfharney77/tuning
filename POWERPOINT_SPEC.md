# PowerPoint Spec — Mets QLoRA Fine-Tuning (tuning)

A spec for a slide deck that explains the `tuning` project.

## Goal & audience
~11 slides / ~9 min for ML engineers. Show a small, measurable QLoRA fine-tune and how learning is
proven (not eyeballed).

## Build notes
16:9; **python-pptx** or hand-author. `slides2video` can narrate it. Source: `MANUAL.md`, `SPEC.md`, `README.md`.

## Slide outline
1. **Title** — "Mets QLoRA Fine-Tuning — teach a small model baseball, then prove it learned."
2. **Goal** — Fine-tune a small, "bare" model to learn New York Mets facts + general baseball rules
   and stats, on a single 8 GB GPU (RTX 4060).
3. **Why this setup** — start from a model that knows little about the Mets so the training effect is
   clearly visible *and measurable*.
4. **QLoRA in one slide** — 4-bit quantized base + low-rank adapters → fits 8 GB, fast to train.
5. **Data** — Mets facts + baseball rules/stats; how the training set is built.
6. **Leakage-free evaluation** — a held-out, **paraphrased** test set scores whether the model
   *generalized* rather than *memorized*.
7. **Comparison UI** — side-by-side base vs. fine-tuned answers for the same prompt.
8. **Results** — scored before/after; show the measured gain.
9. **Hardware/repro** — single 8 GB GPU; what to expect for time/VRAM.
10. **Demo** — ask both models a Mets question; show the score delta.
11. **Closing** — run steps; link `MANUAL.md` (guide) and `SPEC.md` (design rationale).

## Assets to capture
The side-by-side comparison UI, a before/after score chart, and a one-line QLoRA diagram
(quantized base + LoRA adapters).
