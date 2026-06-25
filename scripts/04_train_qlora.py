"""QLoRA fine-tune Qwen2.5-0.5B on the Mets + baseball dataset (fits in 8GB VRAM).

Written for trl >= 1.6 (SFTConfig + processing_class) and transformers >= 5.
"""
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from common import BASE_MODEL, ADAPTER_DIR, DATA_PATH, format_example


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    def fmt(ex):
        return {"text": format_example(ex["instruction"], ex["output"]) + tok.eos_token}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    args = SFTConfig(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print(f"[ok] adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
