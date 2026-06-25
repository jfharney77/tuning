"""Shared helpers for the Mets fine-tuning project."""
import os

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "mets-qlora")
MERGED_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "mets-merged")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mets_qa.jsonl")
QUESTIONS = os.path.join(os.path.dirname(__file__), "..", "eval", "questions.txt")

# Simple Alpaca-style prompt used for both training and inference.
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_example(instruction, output=""):
    return PROMPT.format(instruction=instruction) + output
