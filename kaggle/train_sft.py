"""Kaggle: QLoRA SFT on Qwen3-4B (or Phi-4-mini) for the edge medical model.

Kaggle setup:
    1. Notebook settings: Accelerator = GPU (P100 or T4 is fine), Internet ON
    2. Add your dataset (dataset/final/ as a Kaggle Dataset, or upload train.jsonl)
    3. In a cell first:  !pip install -q unsloth "peft==0.15.2" trl datasets
    4. Run this script with the right --data path

Usage:
    python train_sft.py --data /kaggle/input/medchat-final/train.jsonl \
        --model unsloth/Qwen3-4B-Instruct --out /kaggle/working/sft_qwen3_4b

Expected: ~20k rows, 3 epochs, lr 2e-4 -> roughly 2-3h on a T4.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError as e:
    print("Missing deps, installing unsloth + friends...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "unsloth", "trl", "datasets", "accelerate", "peft"], check=True)
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel, is_bfloat16_supported


def to_messages(conversations):
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    return [{"role": role_map.get(m.get("from", "human"), "user"), "content": m["value"]}
            for m in conversations]


def format_row(tokenizer):
    def fmt(row):
        return {"text": tokenizer.apply_chat_template(
            to_messages(row["conversations"]), tokenize=False, add_generation_prompt=False)}
    return fmt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="/kaggle/input/medchat-final/train.jsonl")
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Instruct")
    ap.add_argument("--out", default="/kaggle/working/sft_qwen3_4b")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--flash", action="store_true", help="try flash attention (Ampere+ only)")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"ERROR: {data_path} not found. Upload dataset/final/ to Kaggle and fix --data.")

    fp16 = not is_bfloat16_supported()  # T4/P100 -> fp16
    print(f"dtype: {'bf16' if not fp16 else 'fp16'}")

    attns = ["flash_attention_2", "sdpa"] if args.flash else ["sdpa"]
    model = tokenizer = None
    for attn in attns:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=args.max_seq_len,
                dtype=None,
                load_in_4bit=True,
                attn_implementation=attn,
            )
            break
        except Exception as e:
            print(f"attn {attn} failed ({e}), trying next")
    if model is None:
        raise SystemExit("model load failed")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = load_dataset("json", data_files=str(data_path))["train"]
    ds = ds.map(format_row(tokenizer), remove_columns=ds.column_names)

    def row_len(row):
        return len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])

    lens = ds.map(lambda r: {"n_tok": row_len(r)})
    max_tok = args.max_seq_len - 32
    dropped = sum(1 for x in lens["n_tok"] if x > max_tok)
    if dropped:
        print(f"WARNING: dropping {dropped} rows longer than {max_tok} tokens "
              f"(avoids silent truncation of JSON labels)")
        ds = lens.filter(lambda r: r["n_tok"] <= max_tok)
    else:
        ds = lens
    print(f"loaded {len(ds)} rows")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            fp16=fp16,
            bf16=not fp16,
            logging_steps=20,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            output_dir="/kaggle/working/ckpt",
            seed=42,
            optim="adamw_8bit",
        ),
    )

    trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(args.out)
    print(f"saved merged model -> {args.out}")


if __name__ == "__main__":
    main()
