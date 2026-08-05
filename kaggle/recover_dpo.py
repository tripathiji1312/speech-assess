"""Kaggle: rebuild the DPO model after a disk-full save crash, without retraining.

The DPO run trains fine but `save_pretrained_merged` can die on a full
/kaggle/working (~19.5GB quota) because sft_qwen3_4b (~8GB) + dpo_ckpt
(~8GB) + the merged output (~8GB) do not fit together. The LoRA adapters
survive, so the final model can be re-merged in two stages:

  stage 1: base (hub) + SFT adapter  -> sft_qwen3_4b   (recreates the SFT model)
  stage 2: sft_qwen3_4b + DPO adapter -> sft_dpo        (final model for Cells 7-9)

The base-model download is redirected to /tmp (~80GB on Kaggle) so the
/kaggle/working quota only ever holds the two 8GB merged outputs.

Usage (in the notebook, after `!git pull`):
    !python /kaggle/working/medchat/kaggle/recover_dpo.py \
        --sft /kaggle/working/ckpt \
        --dpo /kaggle/working/sft_dpo \
        --base Qwen/Qwen3-4B-Instruct-2507 \
        --sft-out /kaggle/working/sft_qwen3_4b \
        --final-out /kaggle/working/sft_dpo

If the SFT adapter is gone too (no checkpoint under /kaggle/working/ckpt),
the SFT weights cannot be recovered and you must re-run Cell 4 + Cell 6.
"""

import os

# Download the 8GB base model to /tmp (~80GB on Kaggle), NOT /kaggle/working.
# Must be set before any transformers/huggingface_hub import.
os.environ.setdefault("HF_HOME", "/tmp/hf_home")

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import torch
    from unsloth import FastLanguageModel
except ImportError:
    print("Missing deps, installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "unsloth==2026.8.2", "transformers==5.5.0", "trl==0.24.0",
                    "peft==0.20.0", "bitsandbytes==0.50.0", "accelerate==1.10.1",
                    "datasets==4.3.0", "xformers==0.0.35", "wandb==0.28.1"], check=True)
    import torch
    from unsloth import FastLanguageModel


def find_adapter_dir(*candidates):
    """Locate a dir containing adapter_config.json + adapter weights, either at
    the candidate root or in a nested checkpoint-NNN subdir. Trainer checkpoints
    (and the mv-misplaced sft_dpo/checkpoint-133) nest one level down."""
    for cand in candidates:
        root = Path(cand)
        if not root.is_dir():
            continue
        for d in [root] + sorted(root.glob("*/")):
            if not (d / "adapter_config.json").is_file():
                continue
            if any(d.glob("adapter_model*.safetensors")) or (d / "adapter_model.bin").is_file():
                return d
    return None


def pin_base_model(adapter_dir, base):
    """Rewrite adapter_config.json's base_model_name_or_path so the loader
    resolves the base we intend (hub id for SFT, local dir for DPO)."""
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    old = cfg.get("base_model_name_or_path")
    if old != base:
        cfg["base_model_name_or_path"] = base
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"[recover] repointed {cfg_path} base: {old} -> {base}")


def load_and_merge(adapter_dir, out_dir, max_seq_len):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        attn_implementation="sdpa",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(out_dir, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(out_dir)
    print(f"[recover] merged -> {out_dir}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft", default="/kaggle/working/ckpt")
    ap.add_argument("--dpo", default="/kaggle/working/sft_dpo")
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--sft-out", default="/kaggle/working/sft_qwen3_4b")
    ap.add_argument("--final-out", default="/kaggle/working/sft_dpo")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()

    sft_adapter = find_adapter_dir(args.sft)
    if sft_adapter is None:
        raise SystemExit(
            f"[recover] SFT adapter not found under {args.sft}. The SFT weights "
            "cannot be recovered from disk - re-run Cell 4 (train_sft.py, ~6h) "
            "then Cell 6 (dpo_stage2.py).")
    dpo_adapter = find_adapter_dir(args.dpo, "/kaggle/working/dpo_ckpt")
    if dpo_adapter is None:
        raise SystemExit(
            f"[recover] DPO adapter not found under {args.dpo} or "
            "/kaggle/working/dpo_ckpt - the DPO weights are lost; re-run Cell 6.")
    print(f"[recover] SFT adapter: {sft_adapter}")
    print(f"[recover] DPO adapter: {dpo_adapter}")

    # The final output dir may currently BE the (misplaced) adapter dir, so
    # stage it elsewhere before we wipe it to write the merged model.
    staging = Path("/kaggle/working/dpo_adapter_staging")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(dpo_adapter, staging)
    pin_base_model(staging, args.sft_out)
    shutil.rmtree(args.final_out, ignore_errors=True)
    if Path(args.sft_out).is_dir():
        shutil.rmtree(args.sft_out, ignore_errors=True)

    print(f"[recover] stage 1: base + SFT adapter -> {args.sft_out}")
    pin_base_model(sft_adapter, args.base)
    load_and_merge(sft_adapter, args.sft_out, args.max_seq_len)

    print(f"[recover] stage 2: SFT model + DPO adapter -> {args.final_out}")
    load_and_merge(staging, args.final_out, args.max_seq_len)

    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(args.sft, ignore_errors=True)
    shutil.rmtree("/kaggle/working/dpo_ckpt", ignore_errors=True)
    print("[recover] done. Adapters no longer needed and were removed.\n"
          "Continue with Cell 5 (eval), Cell 7 (GGUF), Cell 8, Cell 9.\n"
          "If Cell 7 needs room, delete the SFT copy first: !rm -rf /kaggle/working/sft_qwen3_4b")


if __name__ == "__main__":
    main()
