# Kaggle Training — Step-by-Step Runbook

Runs the full pipeline on a free Kaggle GPU:
**SFT (QLoRA Qwen3-4B) → DPO (hallucination reduction) → GGUF Q4_K_M export → on-device smoke test.**

Expected total time on a T4 (free tier): **~4-6 h** (SFT ~3-4 h, DPO ~20-30 min, export ~10 min).
On an L4/A100: roughly half.

---

## 0. Before you start (on your laptop)

### 0.1 Push the repo to GitHub

```bash
cd /home/tripathiji/projects/speech
git init && git add -A && git commit -m "medchat pipeline: data generators, dataset, training, eval"
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

### 0.2 Upload the dataset to Kaggle (recommended)

Option A — Kaggle Dataset:
1. Create a new Dataset at https://www.kaggle.com/datasets (`New Dataset` → upload a folder).
2. Upload a zip containing:
   - `dataset/final/train.jsonl`
   - `dataset/final/val.jsonl`
   - `dataset/dpo.jsonl`
   - `configs/extraction.gbnf`
   - `configs/guardrail.gbnf`
3. Name it e.g. `medchat-final`. Note the slug: `/kaggle/input/medchat-final`.

Option B — skip the Dataset, upload the zip directly to the notebook with the
"Add input → Upload" button. The path will still be `/kaggle/input/<zip-name>/`.

> If you only have the zip in the notebook, replace `medchat-final` in every
> cell below with your actual `/kaggle/input/` folder name.

### 0.3 Verify your data locally (optional but recommended)

```bash
python scripts/make_train_set.py -o dataset/final/     # should print train/val counts
wc -l dataset/dpo.jsonl                                # should print ~1047
python scripts/eval_harness.py --help                  # parses fine
```

---

## 1. Create the Kaggle notebook

1. https://www.kaggle.com → Notebooks → New Notebook.
2. Settings (right panel):
   - **Accelerator: GPU T4 x2** (or P100; L4/A100 if available — faster)
   - **Internet: ON** (required to pip-install unsloth)
3. Add your `medchat-final` dataset: **Add Input → your dataset name**.
4. Save. Then run the cells below **in order**.

> Free-tier GPU quota: a T4 session lasts up to 9 h — enough for the whole
> pipeline in one session. If you run out of time, re-run the notebook
> "Resume previous version" — SFT saves a checkpoint per epoch and picks up
> nothing automatically, but re-running from the SFT cell onward is cheap if
> the SFT model dir still exists in `/kaggle/working` (it does not survive
> session end, so download after each stage if you split sessions).

---

## 2. Cells — copy each block into its own cell

### Cell 1 — Clone the repo

```bash
cd /kaggle/working
git clone https://github.com/<your-user>/<your-repo>.git medchat
cd medchat
ls                                # expect: dataset/ kaggle/ scripts/ configs/ ...
```

### Cell 2 — Install dependencies (one-time, ~2-4 min)

```python
!pip install -q --no-warn-script-location \
    -U unsloth trl datasets accelerate peft bitsandbytes scikit-learn

# optional: on-device smoke test with llama.cpp bindings (used in Cell 7)
!pip install -q --no-warn-script-location llama-cpp-python
```

> If the install fails on `unsloth`, retry once — Kaggle mirrors are flaky.
> Never mix `!pip` and `!apt` for these; unsloth ships its own CUDA deps.

### Cell 3 — Sanity-check the data

```python
import json
from pathlib import Path

data_root = Path("/kaggle/input/medchat-final")
assert (data_root / "train.jsonl").exists(), "dataset not mounted - check Add Input"

for name in ["train.jsonl", "val.jsonl", "dpo.jsonl"]:
    n = sum(1 for _ in open(data_root / name) if _.strip())
    print(f"{name}: {n} rows")

r = json.loads(next(open(data_root / "train.jsonl")))
print("first row _id:", r["_id"], "| roles:", [m["from"] for m in r["conversations"]])
print("sample system:", r["conversations"][0]["value"][:60])
```

Expected:
```
train.jsonl: 41845 rows
val.jsonl: 4224 rows
dpo.jsonl: 1047 rows
```
(If your numbers differ slightly, that's fine — they should be in the same ballpark.)

### Cell 4 — SFT stage (Qwen3-4B QLoRA, ~3-4 h on T4)

```python
!python /kaggle/working/medchat/kaggle/train_sft.py \
    --data /kaggle/input/medchat-final/train.jsonl \
    --model unsloth/Qwen3-4B-Instruct \
    --out /kaggle/working/sft_qwen3_4b \
    --epochs 3 --lr 2e-4 \
    --batch-size 4 --grad-accum 2 --max-seq-len 2048
```

What you should see: `dtype: fp16` on T4/P100 (bf16 on L4/A100+), `loaded ~41845 rows`, then per-epoch logs (`loss` decreasing ~2.0 → ~1.3).

Notes:
- **Out of memory?** lower to `--batch-size 2 --grad-accum 4`. Effective batch stays 8.
- **GPU is L4/A100/H100?** add `--flash` for ~1.5x speed.
- Long rows (>2016 tokens) are **dropped with a warning** — silent truncation would corrupt JSON labels, so we filter instead.
- `save_strategy=epoch` keeps the last 1 checkpoint only (`save_total_limit=1`).

Output: merged 16-bit model in `/kaggle/working/sft_qwen3_4b/`.

### Cell 5 — Quick sanity: run the SFT model on 8 held-out rows

Checks that the chat template + output format actually work before spending time on DPO.

```python
!python /kaggle/working/medchat/scripts/eval_harness.py \
    --checkpoint /kaggle/working/sft_qwen3_4b \
    --split /kaggle/input/medchat-final/val.jsonl \
    --max-examples 8 --max-new-tokens 512
```

Expected: `extraction` rows with valid_json > 0 (even if scores are low pre-DPO).
If `valid_json` is 0/8, stop and check: model path, GPU memory, or rerun Cell 4.

### Cell 6 — DPO stage (hallucination reduction, ~20-30 min)

```python
!python /kaggle/working/medchat/kaggle/dpo_stage2.py \
    --model /kaggle/working/sft_qwen3_4b \
    --data /kaggle/input/medchat-final/dpo.jsonl \
    --out /kaggle/working/sft_dpo \
    --beta 0.1 --lr 5e-5 --epochs 1 --batch-size 4 --max-seq-len 1024
```

Expected: `loaded ~1047 preference pairs`, loss trending down.
Output: merged model in `/kaggle/working/sft_dpo/`.

### Cell 7 — Export GGUF Q4_K_M (~2.6 GB)

```python
!python /kaggle/working/medchat/kaggle/export_gguf.py \
    --model /kaggle/working/sft_dpo \
    --out /kaggle/working/medchat-q4.gguf \
    --quant q4_k_m

!ls -lh /kaggle/working/medchat-q4.gguf
```

### Cell 8 — On-device smoke test (llama.cpp + grammar-constrained JSON)

Proves the GGUF works on CPU with the extraction grammar before you download it.

```python
import json
from llama_cpp import Llama

llm = Llama(
    model_path="/kaggle/working/medchat-q4.gguf",
    n_ctx=2048, n_gpu_layers=0, verbose=False,
    chat_template_kwargs={"enable_thinking": False},
)

grammar = open("/kaggle/input/medchat-final/extraction.gbnf").read()
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
If this prints garbage, do **not** download the GGUF — rerun Cell 4/6.

### Cell 9 — Full eval on the val split (final numbers)

```python
!python /kaggle/working/medchat/scripts/eval_harness.py \
    --checkpoint /kaggle/working/sft_dpo \
    --split /kaggle/input/medchat-final/val.jsonl \
    --max-examples 300 --max-new-tokens 512
```

Record the numbers — these are your release gates:

| Metric | Target |
|---|---|
| extraction valid_json | >= 99% |
| urgency acc | >= 0.90 |
| meds F1 | >= 0.90 |
| guardrail corrupt_recall | >= 0.95 |
| guardrail clean_specificity | >= 0.90 |
| grounded abstain_correct | >= 0.95 |
| safety behavior_ok | >= 0.95 |

### Cell 10 — Download artifacts

```python
from IPython.display import FileLink
FileLink("/kaggle/working/medchat-q4.gguf")   # ~2.6 GB
# also save the merged model if you want 16-bit (8 GB, optional)
# FileLink("/kaggle/working/sft_dpo")
```

Click the link → downloads through the Kaggle UI. The GGUF is your deployable.

---

## 3. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `CUDA out of memory` in Cell 4 | `--batch-size 2 --grad-accum 4`; ensure `--flash` is OFF on T4/P100 |
| `model load failed` | Internet dropped mid-download → re-run cell; or swap `--model Qwen/Qwen3-4B-Instruct` |
| `train.jsonl not found` | Dataset not attached → Settings → Add Input; check the actual `/kaggle/input/<slug>` name |
| `valid_json 0/8` in Cell 5 | Format bug — check Cell 5 output text; it prints raw generations for inspection |
| DPO `KeyError: 'prompt'` | Old trl version → re-run Cell 2 (pip) then restart kernel |
| GGUF smoke test prints JSON + noise | `enable_thinking` not disabled — llama.cpp version too old; `pip install -U llama-cpp-python` |
| Session quota ended mid-run | Re-create notebook, add input, run Cell 2, then continue from the stage you were at |
| Slow training on P100 | Expected; P100 is ~40% slower than T4. Same commands work |

## 4. Hyperparameters (why these)

| Setting | Value | Reason |
|---|---|---|
| base model | Qwen3-4B-Instruct | 4B class SOTA 2025/26, MIT license, strong instruction follow |
| QLoRA r / α | 32 / 64 | good capacity for format tasks; α=2r standard |
| lr / schedule | 2e-4 / cosine | QLoRA default; warmup 5% |
| epochs | 3 | 42k rows ≈ 19.5M tokens → 3 epochs ≈ 58M tokens ≈ 14x model size (rule of thumb) |
| packing | OFF | JSON outputs need clean per-example attention, not concatenation |
| fp16 on T4/P100 | forced | bf16 not supported below Ampere |
| DPO beta / lr / epochs | 0.1 / 5e-5 / 1 | single pass over 1k pairs; avoid preference overfitting |
| max_seq_len 2048 (SFT) | longest row ~1500 | rows beyond 2016 are dropped, never truncated |

## 5. After training

1. Deploy GGUF with llama.cpp / llama_cpp_python on-device:
   - Extraction: `grammar-file configs/extraction.gbnf`
   - Guardrail pass: `grammar-file configs/guardrail.gbnf`, same transcript + draft JSON as input
   - Always set `chat_template_kwargs={"enable_thinking": false}` (thinking was never trained)
2. Real-world checklist before any production use:
   - collect 200-500 **real** transcripts and run them through `eval_harness.py`
   - physician review of ~300 sampled outputs
   - test on slowest target phone (latency, RAM)
