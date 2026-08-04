"""Kaggle: merge LoRA + export GGUF Q4_K_M for llama.cpp on-device deployment.

Usage:
    !pip install -q --no-warn-script-location --extra-index-url \
        https://download.pytorch.org/whl/cu128 \
        "torch==2.10.0+cu128" "unsloth==2026.8.2" "transformers==5.5.0" \
        "trl==0.24.0" "peft==0.20.0" "bitsandbytes==0.50.0" \
        "accelerate==1.10.1" "datasets==4.3.0" "xformers==0.0.35"
    python export_gguf.py --model /kaggle/working/sft_dpo --out /kaggle/working/medchat-q4.gguf
"""

import argparse
import subprocess
import sys

try:
    from unsloth import FastLanguageModel
except ImportError:
    print("installing unsloth...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "unsloth==2026.8.2", "transformers==5.5.0", "trl==0.24.0",
                    "peft==0.20.0", "bitsandbytes==0.50.0", "accelerate==1.10.1",
                    "datasets==4.3.0", "xformers==0.0.35"], check=True)
    from unsloth import FastLanguageModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/kaggle/working/sft_dpo")
    ap.add_argument("--out", default="/kaggle/working/medchat-q4.gguf")
    ap.add_argument("--quant", default="q4_k_m",
                    help="q4_k_m (default, ~2.6GB for 4B), q8_0, f16, q4_0...")
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.for_inference(model)
    gc = getattr(model, "generation_config", None)
    if gc is not None and hasattr(gc, "chat_template_kwargs"):
        gc.chat_template_kwargs = {"enable_thinking": False}
        print("disabled Qwen3 thinking mode (deterministic JSON output)")
    model.save_pretrained_gguf(args.out, tokenizer, quantization_method=args.quant)
    print(f"exported -> {args.out}")
    print("deploy with llama.cpp / llama_cpp_python, use configs/extraction.gbnf for JSON decoding")


if __name__ == "__main__":
    main()
