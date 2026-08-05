# Kaggle Training — Step-by-Step Runbook (exact cells)

Full pipeline on a free Kaggle GPU:
**SFT (QLoRA Qwen3-4B) → DPO (hallucination reduction) → GGUF Q4_K_M export → on-device smoke test.**

Measured time (T4 x2, Aug 2026): **~16 s/step ≈ 12 h per SFT epoch on 1 GPU, ~6 h on 2 GPUs.**
Kaggle caps a GPU session at ~9 h (30 h/week), so a 3-epoch single-GPU run (~36 h) **cannot finish** —
use the Cell 4 2-GPU command with `--epochs 1` (~6 h), then DPO ~20-30 min + export ~10 min.
Every cell is self-contained (absolute paths, no `cd` needed) — copy each block into its own notebook cell exactly as written.

---

## 0. Before you start

Your repo `https://github.com/tripathiji1312/speech-assess` is public and contains code + dataset
(`dataset/final/train.jsonl` 88 MB, `val.jsonl`, `dataset/dpo.jsonl`). No Kaggle Dataset upload needed.

### Create the notebook
1. https://www.kaggle.com → Notebooks → New Notebook.
2. Settings (right panel):
   - **Accelerator: GPU T4 x2** (or P100; L4/A100 if available — faster)
   - **Internet: ON** (required for git clone + pip install)
3. Save. Then run the 10 cells below **in order**.

---

## Cells — copy each block into its own cell

### Cell 1 — Clone the repo (code + dataset in one step)

```bash
!rm -rf /kaggle/working/medchat && cd /kaggle/working && \
  git clone --depth 1 https://github.com/tripathiji1312/speech-assess.git medchat
!ls /kaggle/working/medchat
!du -sh /kaggle/working/medchat/dataset/final/
```

> The `rm -rf` is essential — it guarantees a **fresh copy every run** so stale
> scripts/dataset can't silently keep running the old code.

Expected output:
```
dataset/  kaggle/  scripts/  configs/  KAGGLE_TRAINING.md  PLAN.md  README.md  ...
94M	/kaggle/working/medchat/dataset/final/
```

### Cell 2 — Install dependencies (one-time, ~2-6 min)

```python
# Pinned to the exact versions validated against each other (Aug 2026):
#   unsloth 2026.8.2 + transformers 5.5.0 + trl 0.24.0 (+ peft 0.20.0, bitsandbytes
#   0.50.0, accelerate 1.10.1, datasets 4.3.0, xformers 0.0.35, wandb 0.28.1,
#   torch 2.10.0+cu128).
# Do NOT unpin: e.g. 'pip install -U trl' would pull trl 1.9.2, which unsloth rejects.
!pip install -q --no-warn-script-location --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0+cu128" \
    "unsloth==2026.8.2" \
    "transformers==5.5.0" \
    "trl==0.24.0" \
    "peft==0.20.0" \
    "bitsandbytes==0.50.0" \
    "accelerate==1.10.1" \
    "datasets==4.3.0" \
    "xformers==0.0.35" \
    "wandb==0.28.1" \
    scikit-learn
!pip install -q --no-warn-script-location llama-cpp-python
```

> The `--extra-index-url` is required: the `torch==2.10.0+cu128` wheel only exists on
> the PyTorch cu128 index, not PyPI. (This matches the Kaggle base image — no torch upgrade needed.)
> If the install fails on `unsloth`, just re-run this cell once — Kaggle mirrors are flaky.

### Cell 3 — Sanity-check the data

```python
import json
from pathlib import Path

base = Path("/kaggle/working/medchat")
files = {
    "train.jsonl": base / "dataset/final/train.jsonl",
    "val.jsonl": base / "dataset/final/val.jsonl",
    "dpo.jsonl": base / "dataset/dpo.jsonl",
}
for name, p in files.items():
    print(f"{p}: {sum(1 for _ in open(p) if _.strip())} rows")

r = json.loads(next(open(files["train.jsonl"])))
print("first row _id:", r["_id"], "| roles:", [m["from"] for m in r["conversations"]])
print("sample system:", r["conversations"][0]["value"][:60])
```

Expected output:
```
/kaggle/working/medchat/dataset/final/train.jsonl: 41845 rows
/kaggle/working/medchat/dataset/final/val.jsonl: 4224 rows
/kaggle/working/medchat/dataset/dpo.jsonl: 1047 rows
```

### Cell 4 — SFT stage (Qwen3-4B QLoRA, both T4s, 1 epoch ≈ 6 h)

**Recommended (T4 x2):** torchrun uses both GPUs (effective batch = 4 × 2 GPUs × 2 accum = 16, same as the runbook's original settings). `--epochs 1` is the practical max per session — see the note below.

```bash
!torchrun --standalone --nproc_per_node=2 /kaggle/working/medchat/kaggle/train_sft.py \
    --data /kaggle/working/medchat/dataset/final/train.jsonl \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --out /kaggle/working/sft_qwen3_4b \
    --epochs 1 --lr 2e-4 \
    --batch-size 4 --grad-accum 2 --max-seq-len 2048
```

**1-GPU fallback (P100 or single T4):** same flags via `python`, but ~12 h/epoch — plan one epoch per session:

```bash
!python /kaggle/working/medchat/kaggle/train_sft.py \
    --data /kaggle/working/medchat/dataset/final/train.jsonl \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --out /kaggle/working/sft_qwen3_4b \
    --epochs 1 --lr 2e-4 \
    --batch-size 4 --grad-accum 2 --max-seq-len 2048
```

What you should see: `dtype: fp16` (T4/P100) or `bf16` (L4/A100+), `loaded 41845 rows`, and with torchrun a `Data Parallel GPUs = 2` banner. Unsloth auto-doubles the batch size to fill VRAM, so the step count you see is ~half of the naive `rows / (batch × accum)` estimate — expected, not a bug. Per-epoch loss should decrease (~2.0 → ~1.3).

> **Why `--epochs 1`?** 3 epochs measured at ~36 h — above Kaggle's ~9 h session cap and 30 h/week quota. 1 epoch = ~19.5M tokens ≈ 5x model size, which is enough for format tuning; the Cell 5 eval gate decides if you ever need more. **No resume support** (script calls `trainer.train()` fresh), so a killed session restarts from scratch — a 2-GPU 1-epoch run fits comfortably in one session.

Troubleshoot:
- **Out of memory?** → `--batch-size 2 --grad-accum 4`
- **L4/A100 GPU?** → add `--flash`
- **torchrun not found / spawn errors?** → re-run Cell 2 (torchrun ships with torch 2.10.0+cu128); if it persists, fall back to the 1-GPU command above.

Output: merged 16-bit model in `/kaggle/working/sft_qwen3_4b/`.

### Cell 5 — Quick sanity: run SFT model on 8 held-out rows (format check)

```bash
!python /kaggle/working/medchat/scripts/eval_harness.py \
    --checkpoint /kaggle/working/sft_qwen3_4b \
    --split /kaggle/working/medchat/dataset/final/val.jsonl \
    --max-examples 8 --max-new-tokens 512
```

Expected: an `extraction` section with `valid_json` > 0 (scores will be low pre-DPO — that's fine).
If `valid_json: 0/8`, stop — do not continue; check raw output / re-run Cell 4.

### Cell 6 — DPO stage (hallucination reduction, ~20-30 min)

```bash
!python /kaggle/working/medchat/kaggle/dpo_stage2.py \
    --model /kaggle/working/sft_qwen3_4b \
    --data /kaggle/working/medchat/dataset/dpo.jsonl \
    --out /kaggle/working/sft_dpo \
    --beta 0.1 --lr 5e-5 --epochs 1 --batch-size 4 --max-seq-len 1024 \
    --wandb
```

Expected: `loaded 1047 preference pairs`, loss trending down.
Output: merged model in `/kaggle/working/sft_dpo/`.
The script auto-deletes `/kaggle/working/dpo_ckpt` after training so the ~8GB merged save fits next to the SFT model in `/kaggle/working` (~19.5GB quota).

### Cell 7 — Export GGUF Q4_K_M (~2.6 GB)

```bash
!python /kaggle/working/medchat/kaggle/export_gguf.py \
    --model /kaggle/working/sft_dpo \
    --out /kaggle/working/medchat-q4.gguf \
    --quant q4_k_m
!ls -lh /kaggle/working/medchat-q4.gguf
```

### Cell 8 — On-device smoke test (llama.cpp + grammar-constrained JSON)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="/kaggle/working/medchat-q4.gguf",
    n_ctx=2048, n_gpu_layers=0, verbose=False,
    chat_template_kwargs={"enable_thinking": False},
)

grammar = open("/kaggle/working/medchat/configs/extraction.gbnf").read()
transcript = (
    "Doctor: What's going on today?\n"
    "Patient: chest pain for about 2 hours. It's severe.\n"
    "Doctor: Any other symptoms?\n"
    "Patient: Yes, sweating.\n"
    "Doctor: What medications are you taking?\n"
    "Patient: I take Metformin 500 mg twice daily."
)
sys = ("You are a medical intake assistant. Extract the structured summary "
       "as JSON. Use only information stated in the transcript. "
       'If something was not asked, mark it null/empty and note it in missing_info.')
out = llm.create_chat_completion(
    messages=[{"role": "system", "content": sys},
              {"role": "user", "content": f"Transcript:\n{transcript}"}],
    grammar=grammar, temperature=0.0, max_tokens=512,
    chat_template_kwargs={"enable_thinking": False},
)
print(out["choices"][0]["message"]["content"])
```

Expected: one valid JSON object with `urgency`, `medications`, `missing_info`, etc.
If garbage → do **not** download; re-run Cell 4/6.

### Cell 9 — Full eval on val split (final numbers)

```bash
!python /kaggle/working/medchat/scripts/eval_harness.py \
    --checkpoint /kaggle/working/sft_dpo \
    --split /kaggle/working/medchat/dataset/final/val.jsonl \
    --max-examples 300 --max-new-tokens 512
```

Record these — your release gates:

| Metric | Target |
|---|---|
| extraction valid_json | >= 99% |
| urgency acc | >= 0.90 |
| meds F1 | >= 0.90 |
| guardrail corrupt_recall | >= 0.95 |
| guardrail clean_specificity | >= 0.90 |
| grounded abstain_correct | >= 0.95 |
| safety behavior_ok | >= 0.95 |

### Cell 10 — Upload everything to Weights & Biases

```python
import wandb, json, os, re
from pathlib import Path

# Log in — paste API key when prompted, or pre-set via Kaggle Secrets (see note below)
wandb.login()

run = wandb.init(
    project="medchat-edge",
    name="final-export",
    tags=["export", "eval", "gguf"],
    config={
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "quant": "q4_k_m",
        "sft_epochs": 1,  # set to what you actually ran in Cell 4
        "sft_lr": 2e-4,
        "dpo_beta": 0.1,
        "dpo_epochs": 1,
        "train_rows": 41845,
        "val_rows": 4224,
        "dpo_pairs": 1047,
    },
)

# --- 1. Upload the GGUF model artifact ---
gguf_path = "/kaggle/working/medchat-q4.gguf"
if os.path.exists(gguf_path):
    artifact = wandb.Artifact("medchat-gguf", type="model",
                              description="Qwen3-4B fine-tuned, Q4_K_M quantized for on-device deployment")
    artifact.add_file(gguf_path)
    run.log_artifact(artifact)
    print(f"uploaded GGUF artifact ({os.path.getsize(gguf_path)/1e9:.2f} GB)")
else:
    print(f"WARN: {gguf_path} not found, skipping GGUF upload")

# --- 2. Upload GBNF grammars + configs ---
artifact = wandb.Artifact("medchat-configs", type="config",
                          description="GBNF grammars and hyperparameter configs")
for fname in ["configs/extraction.gbnf", "configs/guardrail.gbnf", "configs/sft.yaml", "configs/dpo.yaml"]:
    p = f"/kaggle/working/medchat/{fname}"
    if os.path.exists(p):
        artifact.add_file(p)
run.log_artifact(artifact)
print("uploaded configs artifact")

# --- 3. Run eval and log metrics ---
import subprocess
eval_out = subprocess.run(
    ["python", "/kaggle/working/medchat/scripts/eval_harness.py",
     "--checkpoint", "/kaggle/working/sft_dpo",
     "--split", "/kaggle/working/medchat/dataset/final/val.jsonl",
     "--max-examples", "300", "--max-new-tokens", "512"],
    capture_output=True, text=True
)
eval_text = eval_out.stdout
if eval_out.returncode != 0:
    print("WARN: eval_harness failed:")
    print(eval_out.stderr[-2000:])
else:
    print(eval_text)

# Parse eval metrics robustly
metrics = {}
current_task = ""
for line in eval_text.split("\n"):
    m = re.match(r"^=== (\w+)", line)
    if m:
        current_task = m.group(1)
        continue
    if current_task and ":" in line:
        key, val = line.strip().split(":", 1)
        key = key.strip()
        val = val.strip()
        try:
            metrics[f"eval/{current_task}/{key}"] = float(val)
        except ValueError:
            # Handle "250/300" format
            parts = val.split("/")
            if len(parts) == 2:
                try:
                    metrics[f"eval/{current_task}/{key}"] = int(parts[0]) / int(parts[1])
                except ValueError:
                    pass

if metrics:
    wandb.log(metrics)
    print(f"\nLogged {len(metrics)} eval metrics to wandb")
else:
    print("WARN: no metrics parsed from eval output")

# --- 4. Upload dataset stats ---
stats_path = "/kaggle/working/medchat/dataset/final/stats.json"
if os.path.exists(stats_path):
    with open(stats_path) as f:
        stats = json.load(f)
    summary = stats.get("_summary", {})
    wandb.log({f"dataset/{k}": v for k, v in summary.items()})

# --- 5. Save raw eval output as text file ---
eval_log_path = "/kaggle/working/eval_results.txt"
with open(eval_log_path, "w") as f:
    f.write(eval_text)
wandb.save(eval_log_path)

run.finish()
print("\nDone! View results at: https://wandb.ai")
```

> **First time?** Get your API key from https://wandb.ai/authorize. On Kaggle, add it as a Secret named `WANDB_API_KEY` and load before this cell:
> ```python
> from kaggle_secrets import UserSecretsClient
> import os
> os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
> ```
>
> **Optional: upload the full merged model (~8 GB).** Only do this if you have time left in your session:
> ```python
> sft_dpo_path = "/kaggle/working/sft_dpo"
> if os.path.isdir(sft_dpo_path):
>     artifact = wandb.Artifact("medchat-sft-dpo-merged", type="model",
>                               description="Merged 16-bit SFT+DPO model")
>     artifact.add_dir(sft_dpo_path)
>     wandb.init(project="medchat-edge", name="upload-merged-model")
>     wandb.log_artifact(artifact)
>     wandb.finish()
> ```

### Cell 11 — Download the GGUF

```python
from IPython.display import FileLink
FileLink("/kaggle/working/medchat-q4.gguf")   # ~2.6 GB, the deployable artifact
```

---

## 3. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `can't open file .../scripts/xxx.py: No such file or directory` | Cell ran before Cell 1, or wrong path. Re-run Cell 1; all cells use absolute `/kaggle/working/medchat/...` paths |
| `dataset/dpo.jsonl: No such file or directory` | Same — the repo (and its dataset) lives at `/kaggle/working/medchat/`, not `/kaggle/working/` |
| `cannot pickle 'ConfigModuleInstance'` | Fixed in current code — SFT now uses `transformers.Trainer` with pre-tokenized data (no trl multiprocess re-tokenization). If you see it, you're running the old `train_sft.py`: `!git pull` and re-run |
| `WARNING: Unsloth should be imported before [trl, transformers, peft]` | Fixed in current code (unsloth imported first). `git pull` to update |
| `warmup_ratio is deprecated` | Fixed — code now computes `warmup_steps` explicitly |
| `Skipping import of cpp extensions ...` | Expected with torch 2.10.0+cu128 (this runbook pins it) and harmless — unsloth's fast kernels need torch >= 2.11; not required here |
| `AttributeError: 'int' object has no attribute 'mean'` (Cell 4, step 0) | unsloth 2026.8.2 + transformers >= 4.54 default `average_tokens_across_devices=True`; on 2 visible GPUs (T4 x2) the loss becomes an int and crashes. Fixed in current code (`average_tokens_across_devices=False`); `git pull` |
| `RuntimeError: Expected to mark a variable ready only once ... marked as ready twice` (Cell 4, step 0) | Reentrant gradient checkpointing is incompatible with DDP on 2 GPUs. unsloth's `patch_peft_model` calls `prepare_model_for_kbit_training(..., use_reentrant=True)`, which binds every `Qwen3DecoderLayer._gradient_checkpointing_func` (transformers 5.5.0 `GradientCheckpointingLayer`) to the reentrant checkpoint — a captured partial, so it bypasses module-attribute wrappers. Fixed in current code: `use_gradient_checkpointing=True` (unpatch smart GC) + `_force_nonreentrant_checkpointing(model)` under torchrun, which (1) wraps `torch.utils.checkpoint.checkpoint` and (2) overwrites `_gradient_checkpointing_func` on every module with a non-reentrant partial (`use_reentrant=False, determinism_check="none"`); expect the log line `[DDP fix] forced non-reentrant checkpointing on 37 module(s)` (1 + 36 layers). `git pull` and re-run. Quick check before the ~6h run: append `--max-steps 3 --logging-steps 1` to the Cell 4 command — expect 3 loss lines, no crash (~10 min). If it then OOMs, use `--batch-size 2 --grad-accum 4` |
| `RuntimeError: Model merge failed ... No such file or directory: '<out>/model-00001-of-00002.safetensors'` (after training) | Both ranks ran `save_pretrained_merged` and raced downloading the original shards into the same output dir. Fixed in current code — merge is rank-guarded to `LOCAL_RANK==0`; `git pull` and re-run |
| `datasets.CastError ... 1 new columns ({'corruption'})` | Fixed — `category`/`corruption` now exist on every row |
| `TypeError: Couldn't cast array of type string to null` (train.jsonl) | Fixed — metadata is now sentinel strings (`"none"`), never null, and the loader declares explicit `Features`; `git pull` and re-run Cell 4 |
| Loss jumps / model learns the prompt | Fixed — data collator now preserves the `-100` prompt mask (`DataCollatorForLanguageModeling` was overwriting labels with input_ids); `git pull` |
| `Trainer.__init__() got an unexpected keyword argument 'tokenizer'` | Fixed — code uses `processing_class` (transformers 5.x) with a `tokenizer=` fallback |
| `HFValidationError: Repo id must be in the form ... /kaggle/working/sft_qwen3_4b` | The checkpoint dir doesn't exist — SFT didn't finish, or a new session wiped `/kaggle/working`. Run Cell 4 fully, then Cell 5. `dpo_stage2.py` now fails fast with a "Model directory not found — re-run the SFT stage" message instead |
| `CUDA out of memory` (Cell 4) | `--batch-size 2 --grad-accum 4`; keep `--flash` OFF on T4/P100 |
| `model load failed` | Internet dropped mid-download → re-run Cell 4; or swap `--model Qwen/Qwen3-4B-Instruct-2507` |
| `valid_json 0/8` (Cell 5) | Format bug → look at raw generation, re-run Cell 4 |
| DPO `KeyError` | trl too old → re-run Cell 2, restart kernel |
| GGUF smoke test prints JSON + noise | llama.cpp too old → `!pip install -U llama-cpp-python`; ensure `enable_thinking=False` |
| Session quota ended mid-run | No resume support — a killed run restarts from scratch. Re-create notebook → Cell 1 + Cell 2 → re-run Cell 4 (use the 2-GPU `--epochs 1` command so it fits in one ~9 h session) |
| Slow training on P100 | Expected (~40% slower than T4); same commands work |

## 4. Hyperparameters (why these)

| Setting | Value | Reason |
|---|---|---|
| base model | Qwen3-4B-Instruct-2507 | 4B-class SOTA, MIT license, strong instruction follow |
| QLoRA r / α | 32 / 64 | good capacity for format tasks; α=2r standard |
| lr / schedule | 2e-4 / cosine | QLoRA default; warmup 5% |
| epochs | 1 (per session) | 42k rows ≈ 19.5M tokens ≈ 5x model size; measured ~6 h/epoch on T4 x2, ~12 h on 1 T4 — 3 epochs (~36 h) exceed Kaggle's ~9 h session cap. No resume support, so each session starts fresh; re-run Cell 4 with more epochs only if the Cell 5 gate fails |
| packing | OFF | JSON outputs need clean per-example attention, not concatenation |
| SFT engine | transformers.Trainer (pre-tokenized) | avoids trl's multiprocess pickling crash; prompt masked (loss only on model output) |
| fp16 on T4/P100 | forced | bf16 unsupported below Ampere |
| DPO beta / lr / epochs | 0.1 / 5e-5 / 1 | single pass over 1k pairs; avoid preference overfitting |
| max_seq_len 2048 (SFT) | longest row ~1500 | rows > 2016 tokens are dropped, never truncated |

## 5. After training

1. Deploy GGUF with llama.cpp / llama_cpp_python on-device:
   - Extraction: `grammar-file configs/extraction.gbnf`
   - Guardrail pass: `grammar-file configs/guardrail.gbnf`
   - Always set `chat_template_kwargs={"enable_thinking": false}` (thinking was never trained)
2. Real-world checklist before production:
   - collect 200-500 **real** transcripts, run through `eval_harness.py`
   - physician review of ~300 sampled outputs
   - latency/RAM test on slowest target phone
