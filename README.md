# Edge Medical Chat-Analysis Model

Fine-tuned Qwen3-4B for on-device clinical transcript analysis: extraction, triage, and grounded QA.
Dataset is generated procedurally (no API keys needed) and training runs free on Kaggle.

## Quickstart (local, no GPU/API needed)

```bash
# 1. generate extraction data (transcripts -> gold JSON)
python scripts/transcript_simulator.py --n-train 20000 --n-val 2000 -o dataset/

# 2. build grounded QA from YOUR Mayo/WebMD corpus
#    (drop symptoms.jsonl / drugs.jsonl into data/raw/ first — same format as your sample)
python scripts/build_grounded_qa.py data/raw/*.jsonl -o dataset/grounded_qa.jsonl --o-dpo dataset/dpo.jsonl

# 3. claim-support guardrail set (catches invented fields in the extraction draft)
python scripts/build_guardrail.py --n-train 15000 --n-val 1500 -o dataset/

# 4. safety/refusal set
python scripts/build_safety_set.py -o dataset/safety.jsonl

# 5. assemble ChatML train/val
python scripts/make_train_set.py -o dataset/final/
```

## Train on Kaggle

**Full cell-by-cell runbook: see [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md).**

1. Upload `dataset/final/` (train.jsonl, val.jsonl) + `dataset/dpo.jsonl` as a Kaggle Dataset.
2. Open a Kaggle Notebook: Accelerator = GPU (P100/T4), Internet = ON, add the dataset.
3. Clone this repo inside the notebook, install deps, run the SFT → DPO → GGUF cells.

## Eval

```bash
python scripts/eval_harness.py --checkpoint /kaggle/working/sft_dpo --split dataset/final/val.jsonl --max-examples 300
```

Targets after DPO: extraction urgency acc >= 0.9, valid-JSON >= 99%, meds F1 >= 0.9,
claim-support >= 0.8, abstain-correct >= 0.95, safety behavior-ok >= 0.95.
Guardrail: corrupt-recall >= 0.95, clean-specificity >= 0.90, evidence-rate >= 0.85.

## Pipeline (all on device)

1. **Extraction** — chat transcript → structured JSON (urgency, meds, red flags, vitals, missing info).
2. **Guardrail** — the JSON draft is re-checked against the transcript; every field must trace to an
   exact quote, otherwise the field is flagged with a severity (Abridge-style confabulation elimination).
3. **Triage** — escalate if urgency is emergency/urgent; suggest ER/crisis resources when needed.
4. **Grounded QA** — retrieve from the on-device Mayo/WebMD KB, answer with `Source:` citations, or abstain.

## Repo layout

```
PLAN.md                    build plan + architecture
dataset/                   generated data (train/val, guardrail, safety, grounded QA)
dataset/seed_safety.jsonl  hand-written safety examples (edit freely)
examples/                  demo corpus in your exact jsonl format (replace with real data)
scripts/                   data generation + eval (pure Python, stdlib only)
kaggle/                    training scripts (Unsloth, run on Kaggle)
configs/                   GBNF grammars (extraction + guardrail) + hyperparameter reference
```

## Rules for real-world use

- Model never diagnoses or doses; it extracts, triages, and cites the KB.
- Emergencies escalate to "seek care now"; uncertainty → "consult a doctor."
- No PHI in training data; simulator masks PII with placeholders.
- Physician review of eval outputs before any deployment.
