# Kaggle Training — Step-by-Step Runbook (exact cells)

Full pipeline on a free Kaggle GPU:
**SFT (QLoRA Qwen3-4B) → DPO (hallucination reduction) → GGUF Q4_K_M export → on-device smoke test.**

Expected time on a T4 (free tier): **~4-6 h total** (SFT ~3-4 h, DPO ~20-30 min, export ~10 min).
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
!cd /kaggle/working && git clone https://github.com/tripathiji1312/speech-assess.git medchat
!ls /kaggle/working/medchat
!du -sh /kaggle/working/medchat/dataset/final/
```

Expected output:
```
dataset/  kaggle/  scripts/  configs/  KAGGLE_TRAINING.md  PLAN.md  README.md  ...
94M	/kaggle/working/medchat/dataset/final/
```

### Cell 2 — Install dependencies (one-time, ~2-4 min)

```python
!pip install -q --no-warn-script-location -U unsloth trl datasets accelerate peft bitsandbytes scikit-learn
!pip install -q --no-warn-script-location llama-cpp-python
```

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

### Cell 4 — SFT stage (Qwen3-4B QLoRA, ~3-4 h on T4)

```bash
!python /kaggle/working/medchat/kaggle/train_sft.py \
    --data /kaggle/working/medchat/dataset/final/train.jsonl \
    --model unsloth/Qwen3-4B-Instruct \
    --out /kaggle/working/sft_qwen3_4b \
    --epochs 3 --lr 2e-4 \
    --batch-size 4 --grad-accum 2 --max-seq-len 2048
```

What you should see: `dtype: fp16` (T4/P100) or `bf16` (L4/A100+), `loaded 41845 rows`, per-epoch loss decreasing (~2.0 → ~1.3).

Troubleshoot:
- **Out of memory?** → `--batch-size 2 --grad-accum 4`
- **L4/A100 GPU?** → add `--flash`

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
    --beta 0.1 --lr 5e-5 --epochs 1 --batch-size 4 --max-seq-len 1024
```

Expected: `loaded 1047 preference pairs`, loss trending down.
Output: merged model in `/kaggle/working/sft_dpo/`.

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

### Cell 10 — Download the GGUF

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
| `CUDA out of memory` (Cell 4) | `--batch-size 2 --grad-accum 4`; keep `--flash` OFF on T4/P100 |
| `model load failed` | Internet dropped mid-download → re-run Cell 4; or swap `--model Qwen/Qwen3-4B-Instruct` |
| `valid_json 0/8` (Cell 5) | Format bug → look at raw generation, re-run Cell 4 |
| DPO `KeyError` | trl too old → re-run Cell 2, restart kernel |
| GGUF smoke test prints JSON + noise | llama.cpp too old → `!pip install -U llama-cpp-python`; ensure `enable_thinking=False` |
| Session quota ended mid-run | Re-create notebook → run Cell 1 + Cell 2 → continue from the next stage; re-download model outputs you already have |
| Slow training on P100 | Expected (~40% slower than T4); same commands work |

## 4. Hyperparameters (why these)

| Setting | Value | Reason |
|---|---|---|
| base model | Qwen3-4B-Instruct | 4B-class SOTA, MIT license, strong instruction follow |
| QLoRA r / α | 32 / 64 | good capacity for format tasks; α=2r standard |
| lr / schedule | 2e-4 / cosine | QLoRA default; warmup 5% |
| epochs | 3 | 42k rows ≈ 19.5M tokens × 3 ≈ 58M ≈ 14x model size (rule of thumb) |
| packing | OFF | JSON outputs need clean per-example attention, not concatenation |
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
