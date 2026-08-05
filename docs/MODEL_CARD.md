# Model Card: medchat — Medical Intake Summarizer (Qwen3-4B, QLoRA SFT + DPO)

## 1. Identity

| | |
|---|---|
| Model name | `medchat` |
| Task | Structured clinical summary extraction from patient phone-chat transcripts (strict JSON out) |
| Base model | `Qwen/Qwen3-4B-Instruct-2507` (vocab 151,936) |
| Training | 2-stage QLoRA: **SFT** (JSON extraction) then **DPO** (grounded-answer preference, hallucination reduction) |
| Final artifact | 16-bit merged weights (~7.6 GB) + GGUF Q4_K_M (`medchat-q4.gguf`, ~2.6 GB) |
| License/use | Research/demo only — trained on synthetic data, **not clinically validated** |

## 2. Architecture

- Decoder-only transformer, ~4B params: 36 layers, GQA, SwiGLU FFN, RMSNorm, RoPE, sdpa attention.
- Native context 262,144; **trained at 2,048** (SFT) / 1,024 (DPO) tokens.
- QLoRA: base weights in 4-bit NF4; LoRA on all attention + MLP projections (`q,k,v,o,gate,up,down_proj`).
- dtype: fp16 on T4/P100 (`is_bfloat16_supported()` is False); bf16 on A100.
- Seed 42 everywhere.

## 3. Training configuration

### Stage 1 — SFT
| Hyperparameter | Value |
|---|---|
| Data | 41,845 examples; 3 turns per example: system / user transcript / assistant JSON |
| LoRA rank / alpha / dropout | 32 / 64 / 0.0 |
| Learning rate / schedule | 2e-4, cosine |
| Warmup | 5% (ratio) |
| Epochs | 1 (2,616 steps) |
| Per-device batch / grad accum | 4 / 2 (effective 16) |
| Max sequence length | 2,048 |
| Optimizer | adamw_8bit |
| Packing | false (never pack — JSON correctness needs full attention) |
| Loss masking | instruction turns masked (`labels = -100`) |
| Runtime | ~6 h on 2×T4 |

### Stage 2 — DPO
| Hyperparameter | Value |
|---|---|
| Data | 1,047 preference pairs (chosen = grounded answer with citation; rejected = answer swapped from an unrelated topic, i.e. hallucination trap) |
| beta | 0.1 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.0 |
| Learning rate / schedule | 5e-5, cosine |
| Warmup | 10% of steps (13 steps) |
| Epochs | 1 (133 steps) |
| Per-device batch / grad accum | 4 / 2 |
| Max length / max prompt length | 1,024 / 896 |
| Ref model | None (implicit reference = frozen base) |
| Max grad norm | 1.0 |
| Optimizer | adamw_8bit |
| Filtering | pairs where prompt+chosen or prompt+rejected exceed 1,024−32 tokens are dropped |

### Data format (input to both stages)
SFT example:
```json
{
  "_id": "...",
  "category": "none|emergency|dosing|diagnosis|self_harm|out_of_scope|small_talk|illegal",
  "corruption": "none",
  "conversations": [
    {"from": "system", "value": "You are a medical intake assistant ..."},
    {"from": "human",  "value": "transcript text"},
    {"from": "gpt",    "value": "{\"chief_complaint\": ...}"}
  ]
}
```

DPO example:
```json
{
  "system": "You are a health information assistant. Answer using ONLY the provided context...",
  "prompt": "Context:\n<context>\n\nQuestion: <question>",
  "chosen": "Answer: <grounded answer>\nSource: <source>",
  "rejected": "Answer: <unrelated-topic answer>"
}
```

Dataset split: train 41,845 / val 4,224 rows. Val category mix: none 4,114 · emergency 37 · out_of_scope 29 · dosing 18 · diagnosis 18 · self_harm 5 · small_talk 2 · illegal 1.

## 4. Output schema (strict JSON, these exact keys)

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

System prompt used at inference:
> You are a medical intake assistant running on a patient's phone. Extract the structured clinical summary from the chat transcript below. Only include information that is actually present in the transcript. If something was not discussed, put it in missing_info. urgency must be one of: "routine", "urgent", "emergency". escalate is true unless urgency is routine. Output strict JSON only, with exactly these keys: chief_complaint, hpi, vitals, medications, allergies, red_flags, missing_info, urgency, escalate. Do not output any text outside the JSON.

## 5. Evaluation (final DPO model, 300-row val sample, greedy decode)

| Metric | Definition | Result |
|---|---|---|
| `valid_json` | output parses as JSON after stripping text outside the outermost braces | **300/300** |
| `urgency_acc` | 3-class accuracy (emergency/urgent/routine) | **1.000** |
| `escalate_acc` | binary accuracy; = balanced accuracy (sensitivity = specificity = 1.0 on this set) | **1.000** |
| meds / allergies / red_flags F1 | token-set F1 per list field | **1.000** |
| chief_complaint F1 | token F1 | **1.000** |
| `missing_info_precision` | correct∩predicted / predicted | **0.975** |

Interpretation: benchmark-perfect on this set — but val transcripts are synthetic and
**state the answer verbatim**, so these numbers are an upper bound on pipeline
correctness, **not** clinical accuracy. Validate against real-world transcripts
before any production use.

## 6. Deployment / inference

### GGUF (llama.cpp / llama_cpp_python)
- File: `medchat-q4.gguf` (Q4_K_M, ~2.6 GB). An f16 GGUF is also available if higher fidelity is needed.
- Strict JSON via llama-cli:
  ```
  llama-cli -m medchat-q4.gguf --grammar configs/extraction.gbnf \
    --chat-template qwen3 -p "<system>...</system><user>...transcript...</user>"
  ```
- `llama_cpp_python` chat API notes:
  - Custom GBNF grammar is not applied faithfully via `create_chat_completion` (keys come out unquoted).
  - `response_format={"type": "json_object"}` returns `{}`.
  - **Use plain generation** and validate/repair the JSON, or use llama-cli with the grammar for production.

### HF transformers (16-bit or 4-bit)
- Chat template: Qwen3; set `enable_thinking=False` for the JSON task.
- Generation: greedy (`do_sample=False`), stop at EOS. The model may emit a leading
  `<think>\n\n</think>` block — strip it and any text outside the JSON braces before parsing.

## 7. Grammar files (`configs/`)

- `extraction.gbnf` — strict JSON grammar for the output schema above (llama-cli `--grammar`).
- `guardrail.gbnf` — JSON grammar for the claim-support verifier output.
- Grammar authoring gotcha (llama.cpp parser): one rule per line; rule names must not
  contain `_` (allowed chars: a-z, A-Z, `-`, 0-9) — e.g. `nullableString`, `nullableInt`.

## 8. Reproduce

```python
# Stage 1 SFT (unsloth + transformers.Trainer)
from unsloth import FastLanguageModel, is_bfloat16_supported
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", max_seq_length=2048,
    load_in_4bit=True, dtype=None, attn_implementation="sdpa")
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=64, lora_dropout=0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)
# Trainer: lr 2e-4 cosine, warmup_ratio 0.05, 1 epoch, batch 4, grad_accum 2,
# fp16/bf16 per hardware, adamw_8bit, labels=-100 on instruction tokens

# Stage 2 DPO (trl DPOTrainer)
model = FastLanguageModel.get_peft_model(
    sft_model, r=16, lora_alpha=32, lora_dropout=0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)
# DPOConfig: beta 0.1, lr 5e-5 cosine, 1 epoch, batch 4, grad_accum 2,
# max_length 1024, max_prompt_length 896, ref_model=None, max_grad_norm 1.0, adamw_8bit

# Merge to 16-bit and export
model.save_pretrained_merged("medchat-16bit", tokenizer, save_method="merged_16bit")
# GGUF: unsloth FastLanguageModel.save_pretrained_gguf -> f16,
# then convert_hf_to_gguf.py + quantize Q4_K_M
```
