"""Kaggle: DPO stage 2 on the grounded-QA preference pairs.

Trains the SFT model to prefer grounded answers (with citation) over
ungrounded/hallucinated ones. This is the main hallucination reducer.

Kaggle setup:
    !pip install -q unsloth "peft==0.15.2" trl datasets
    python dpo_stage2.py --model /kaggle/working/sft_qwen3_4b \
        --data /kaggle/input/medchat-final/dpo.jsonl --out /kaggle/working/sft_dpo
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402 - before trl
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig
except ImportError:
    print("Missing deps, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "unsloth", "trl", "datasets", "accelerate", "peft"], check=True)
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/kaggle/working/sft_qwen3_4b")
    ap.add_argument("--data", default="/kaggle/input/medchat-final/dpo.jsonl")
    ap.add_argument("--out", default="/kaggle/working/sft_dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", default="medchat-edge")
    args = ap.parse_args()

    report_to = "none"
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name="dpo-hallucination-reduction",
                   config=vars(args), tags=["dpo", "grounding"])
        report_to = "wandb"

    fp16 = not is_bfloat16_supported()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    raw = [json.loads(l) for l in open(args.data) if l.strip()]

    def msgs(system, text):
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    def n_tok(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    pairs = []
    dropped = 0
    for r in raw:
        if n_tok(r["prompt"]) + n_tok(r["chosen"]) > args.max_seq_len - 32 or \
           n_tok(r["prompt"]) + n_tok(r["rejected"]) > args.max_seq_len - 32:
            dropped += 1
            continue
        pairs.append({
            "prompt": msgs(r["system"], r["prompt"]),
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        })
    if dropped:
        print(f"WARNING: dropped {dropped} pairs exceeding max_seq_len (avoid truncation)")
    ds = Dataset.from_list(pairs)
    print(f"loaded {len(ds)} preference pairs")

    steps = max(1, len(ds) * args.epochs / (args.batch_size * 2))
    warmup_steps = int(steps * 0.1)

    import inspect
    dpo_params = inspect.signature(DPOTrainer.__init__).parameters

    dpo_config = DPOConfig(
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
            fp16=fp16,
            bf16=not fp16,
            warmup_steps=warmup_steps,
            logging_steps=10,
        report_to=report_to,
        output_dir="/kaggle/working/dpo_ckpt",
        seed=42,
        optim="adamw_8bit",
        max_length=args.max_seq_len,
        max_prompt_length=args.max_seq_len - 128,
    )

    dpo_kwargs = dict(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=ds,
    )
    if "max_length" in dpo_params:
        dpo_kwargs["max_length"] = args.max_seq_len
    if "max_prompt_length" in dpo_params:
        dpo_kwargs["max_prompt_length"] = args.max_seq_len - 128
    if "processing_class" in dpo_params:
        dpo_kwargs["processing_class"] = tokenizer
    else:
        dpo_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**dpo_kwargs)

    trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(args.out)
    print(f"saved DPO model -> {args.out}")

    if args.wandb:
        import wandb
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
