"""Kaggle: QLoRA SFT on Qwen3-4B (or Phi-4-mini) for the edge medical model.

Kaggle setup:
    1. Notebook settings: Accelerator = GPU (P100 or T4 is fine), Internet ON
    2. Clone the repo (code + dataset) into /kaggle/working/medchat
    3. Cell:  !pip install -q -U unsloth trl datasets accelerate peft bitsandbytes
    4. Run this script with the right --data path

Usage:
    python train_sft.py --data /kaggle/working/medchat/dataset/final/train.jsonl \
        --model unsloth/Qwen3-4B-Instruct --out /kaggle/working/sft_qwen3_4b

Why plain transformers.Trainer instead of trl.SFTTrainer:
  - trl's SFTTrainer re-tokenizes the dataset internally with num_proc>=2 and
    pickles the mapping function -> 'cannot pickle ConfigModuleInstance' crash
    on Kaggle. We pre-tokenize once, in-process, with the prompt masked out
    (labels=-100 on the instruction part) - faster and no crash.
  - import order: unsloth MUST be imported before trl/transformers/peft.

Expected: ~42k rows, 3 epochs, lr 2e-4 -> roughly 3-4h on a T4.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402 - must be before trl/transformers
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
except ImportError:
    print("Missing deps, installing unsloth + friends...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "unsloth", "trl", "datasets", "accelerate", "peft"], check=True)
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import load_dataset
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def to_messages(conversations):
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    return [{"role": role_map.get(m.get("from", "human"), "user"), "content": m["value"]}
            for m in conversations]


def format_row(tokenizer):
    def fmt(row):
        return {"text": tokenizer.apply_chat_template(
            to_messages(row["conversations"]), tokenize=False, add_generation_prompt=False)}
    return fmt


def make_tokens(tokenizer):
    """Tokenize the full example; mask everything before the assistant turn
    (labels=-100) so loss is only on the model's own output."""
    def fn(row):
        msgs = to_messages(row["conversations"])
        full = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False)
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=True, add_generation_prompt=True)
        ids = full + [tokenizer.eos_token_id]
        prompt_len = min(len(prompt), len(ids))
        labels = [-100] * prompt_len + ids[prompt_len:]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="/kaggle/working/medchat/dataset/final/train.jsonl")
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
        raise SystemExit(f"ERROR: {data_path} not found. Fix --data.")

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
    ds = ds.map(make_tokens(tokenizer), remove_columns=ds.column_names)

    max_tok = args.max_seq_len - 2
    dropped = sum(1 for r in ds if len(r["input_ids"]) > max_tok)
    if dropped:
        print(f"WARNING: dropping {dropped} rows longer than {max_tok} tokens "
              f"(avoids silent truncation of JSON labels)")
        ds = ds.filter(lambda r: len(r["input_ids"]) <= max_tok)
    print(f"loaded {len(ds)} tokenized rows")

    steps_per_epoch = max(1, len(ds) / (args.batch_size * args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"~{total_steps} steps total, warmup {warmup_steps}")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
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
            dataloader_num_workers=2,
        ),
    )

    trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(args.out)
    print(f"saved merged model -> {args.out}")


if __name__ == "__main__":
    main()
