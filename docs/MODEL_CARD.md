# Model Card: medchat — Medical Intake Summarizer (Qwen3-4B, QLoRA SFT + DPO)

## 1. Identity

| | |
|---|---|
| Model name | `medchat` (working dirs: `sft_qwen3_4b` → `sft_dpo`) |
| Task | Structured clinical summary extraction from patient phone-chat transcripts (JSON out) |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (vocab 151,936) |
| Training | 2-stage QLoRA: **SFT** (JSON extraction) then **DPO** (grounded-answer preference, hallucination reduction) |
| Final artifact | 16-bit merged weights (`/kaggle/working/sft_dpo`, ~7.6 GB) + GGUF Q4_K_M (`medchat-q4.gguf`, ~2.6 GB) |
| License/use | Research/demo only — synthetic data, **not clinically validated** |

## 2. Architecture

- Decoder-only transformer, ~4B params: 36 layers, GQA, SwiGLU FFN, RMSNorm, RoPE, sdpa attention.
- Native context 262,144; **trained at 2,048** (SFT) / 1,024 (DPO) tokens.
- QLoRA: base weights in 4-bit NF4; LoRA on all attention + MLP projections.
  - SFT: `r=32, alpha=64, dropout=0, target=[q,k,v,o,gate,up,down]_proj`
  - DPO: `r=16, alpha=32, dropout=0, same targets`
- fp16 on T4 (bf16 unsupported), `adamw_8bit`, seed 42.
- Input format: Qwen3 chat template, system prompt = medical intake assistant; `enable_thinking=False` pinned on tokenizer (see §7 caveat).

## 3. Training

### Stage 1 — SFT (`kaggle/train_sft.py`)
| | |
|---|---|
| Data | `dataset/final/train.jsonl` — 41,845 rows; 3 turns: system / human transcript / gpt JSON |
| Hyperparams | lr 2e-4 cosine, warmup 0.05, 1 epoch, batch 4 × 2 GPU, grad-accum 2 (eff. 16), `max_seq_len 2048` |
| Runs | 2,616 steps, ~6 h on 2×T4; plain `transformers.Trainer` (trl `SFTTrainer` crashes on pickle of `ConfigModuleInstance`) |
| Note | `packing=false` (JSON correctness needs full attention); pre-tokenized once, instruction tokens masked (`labels=-100`) |

### Stage 2 — DPO (`kaggle/dpo_stage2.py`)
| | |
|---|---|
| Data | `dataset/dpo.jsonl` — 1,062 pairs → 1,047 after truncation filter (prompt+response ≤ 1,024 − 32) |
| Definition | `chosen` = answer grounded in provided context (with citation); `rejected` = answer swapped from an unrelated topic (hallucination trap) |
| Hyperparams | `beta 0.1`, lr 5e-5 cosine, 1 epoch, batch 4 × 2, `max_seq_len 1024`, warmup 10% (13 steps), 133 steps, `ref_model=None` (implicit ref = frozen base), `max_grad_norm 1.0` |
| Output | merged 16-bit to `sft_dpo`; checkpoints deleted to fit /kaggle/working |

### Recovery path (used 2026-08-04)
`kaggle/recover_dpo.py`: rebuilds `sft_dpo` from the surviving DPO LoRA adapter (`dpo_adapter_staging`) + SFT base (`/tmp/sft_qwen3_4b`) when the original run ran out of disk mid-save. Stage-2 merges are idempotent against a pre-existing `sft_dpo` dir.

## 4. Data & output schema

- Rows: `{_id, category, corruption, conversations: [system, human, gpt]}`.
- Category mix (val, 4,224 rows): none 4,114 · emergency 37 · out_of_scope 29 · dosing 18 · diagnosis 18 · self_harm 5 · small_talk 2 · illegal 1.
- Output JSON (`dataset/schema.json`):

```json
{
  "chief_complaint": "string",
  "hpi": "string",
  "vitals": {"temp": "string|null", "bp": "string|null", "hr": "int|null", "spo2": "int|null"},
  "medications": ["{name, dose, frequency}"],
  "allergies": ["string"],
  "red_flags": ["string"],
  "missing_info": ["string"],
  "urgency": "emergency|urgent|routine",
  "escalate": "bool  (true iff urgency != routine)"
}
```

## 5. Evaluation (final DPO model, 300-row val sample, greedy)

| Metric | Definition | Result |
|---|---|---|
| `valid_json` | output parses as JSON after `strip_outside_braces` | **300/300** |
| `urgency_acc` | 3-class accuracy (emergency/urgent/routine) | **1.000** |
| `escalate_acc` | binary accuracy; = balanced accuracy (sens=spec=1.0 on this set) | **1.000** |
| meds / allergies / red_flags F1 | token-set F1 per list field | **1.000** |
| chief_complaint F1 | token F1 | **1.000** |
| `missing_info_precision` | correct∩predicted / predicted | **0.975** |

Interpretation: benchmark-perfect on this set — but val transcripts are synthetic and
**state the answer verbatim**, so these numbers are an upper bound on pipeline
correctness, **not** clinical accuracy. Additional harnesses exist in
`scripts/eval_harness.py` for other suites: `eval_grounded` (abstain/source/claim-support),
`eval_safety` (per-category keyword behavior), `eval_guardrail` (claim-support verifier).

## 6. Deployment

| Artifact | Path |
|---|---|
| 16-bit merged | `/kaggle/working/sft_dpo` (~7.6 GB) |
| GGUF Q4_K_M | `/kaggle/working/sft_dpo_gguf/sft_dpo.Q4_K_M.gguf` → copied to `medchat-q4.gguf` (~2.6 GB) |
| LoRA adapters (archive) | `/kaggle/working/ckpt/checkpoint-2616` (SFT), `/kaggle/working/dpo_adapter_staging` (DPO) |

- Strict JSON in **llama-cli**: `--grammar configs/extraction.gbnf` (one rule per line; rule names must not contain `_`).
- `llama_cpp_python` chat API limitations: custom GBNF yields unquoted keys, `response_format={"type":"json_object"}` returns `{}` → use plain generation and validate/post-process, or llama-cli for production.
- GGUF export: `kaggle/export_gguf.py` (unsloth f16 → `convert_hf_to_gguf.py`, quant Q4_K_M; `--out` target name honored by copying the produced file).

## 7. Known caveats / gotchas

1. Model emits a leading `<think>\n\n</think>` block; harness strips it. `enable_thinking=False` is only honored by chat-template consumers (llama-cli), not `model.generate`.
2. `transformers 5.5` tokenizer `pad()` raises on `BatchEncoding` dicts → `eval_harness.py` pads manually, left-padded (decoder-only).
3. `/kaggle/working` is wiped between sessions; ~19.5 GB total → SFT base + checkpoints + merged output do not fit simultaneously (delete checkpoints before `save_pretrained_merged`).
4. Training has no resume support; 1 epoch / stage kept under Kaggle's 9 h cap.
5. Benign warnings: `fix_mistral_regex` Qwen3 false positive (do not set), cpp-extension torch version mismatch, `max_new_tokens` overrides `max_length`.

## 8. Reproduce

```bash
# SFT (2×T4, ~6 h)
python kaggle/train_sft.py --data dataset/final/train.jsonl --out /kaggle/working/sft_qwen3_4b
# DPO
python kaggle/dpo_stage2.py --model /kaggle/working/sft_qwen3_4b --data dataset/dpo.jsonl --out /kaggle/working/sft_dpo
# Eval
python scripts/eval_harness.py --model /kaggle/working/sft_dpo --split dataset/final/val.jsonl
# Export
python kaggle/export_gguf.py --model /kaggle/working/sft_dpo --out medchat-q4.gguf
```

Config references: `configs/sft.yaml`, `configs/dpo.yaml`, `configs/extraction.gbnf`, `configs/guardrail.gbnf`.
