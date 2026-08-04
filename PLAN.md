# Edge Medical LLM — Build Plan

**Goal:** On-device model (mobile) that takes chat transcripts, extracts structured clinical info,
triage/escalation decisions, and answers with grounded, cited info. Trained on Kaggle (free).

**Current state:** sub-2B models fine-tuned on encyclopedia-style QA (Mayo/WebMD) — task mismatch +
no eval = bad results. This plan fixes the data, the model, and the eval.

---

## Architecture (final form)

```
                    ┌────────────────────────── on device ──────────────────────────┐
 raw chat transcript │  Pass 1: extract structured JSON (grammar-constrained decode)  │
  ─────────────────►  │  Pass 2: claim-support guardrail: every extracted claim must    │
    (app captures)    │          trace to a transcript quote, else flag the field       │
                      │  Pass 3: triage (urgency, red flags, escalate?)                 │
                      │  Pass 4: grounded QA: retrieve from on-device KB → cite/abstain │
                     └───────────────┬──────────────────────────────────────────────────┘
                                     │ only de-identified JSON leaves the phone
                           ┌─────────▼──────────┐
                           │ cloud (optional):  │  heavy reasoning: differentials,
                           │ big model          │  drug interactions, full review
                           └────────────────────┘
```

Small model = **parser + triager + grounded answerer**, NOT a medical knowledge store.
The Mayo/WebMD corpus you already have becomes the **retrieval KB**, not the training labels.

---

## Data (what we build, in order)

| Split | Size (target) | Source | Generator | Labels |
|---|---|---|---|---|
| A. extraction | 20k train / 2k val | synthetic messy transcripts | `transcript_simulator.py` (no API) | gold JSON from simulator ground truth |
| A2. guardrail | 15k train / 1.5k val | A, procedurally corrupted | `build_guardrail.py` | issue list w/ severity + transcript quote |
| B. grounded QA | 10-20k | your Mayo/WebMD jsonl (symptoms.jsonl, drugs.jsonl) | `build_grounded_qa.py` | answer + source citation, abstain cases |
| C. safety/refusal/emergency | 3-5k | templates + seed | `build_safety_set.py` + `dataset/seed_safety.jsonl` | escalation/refusal labels |
| D. preference pairs (DPO) | 5-10k | procedural negatives from B | `build_grounded_qa.py --dpo` | chosen/rejected |
| E. (optional) paraphrased transcripts | 10-20k | teacher model on Kaggle (Qwen3-30B-A3B, free GPU) | `distill_paraphrase_on_kaggle.py` | improves realism |

**Key design decisions**
- Labels for A are *guaranteed correct* (the simulator knows the ground truth it injected).
  No teacher API required — everything runs on a laptop or Kaggle CPU/GPU free tier.
- B uses *your existing data* as evidence: (context chunk + question) → answer **with citation**,
  or "the source doesn't cover this" → trains grounding + abstention, kills memorization-style hallucination.
- Every example that could confuse the model (missing info, conflicting info, PHI) is explicitly modeled.
- A2 is the confabulation-elimination stage (Abridge whitepaper): corrupt exactly one field of the
  gold extraction (invented meds/vitals/allergies, swapped doses, flipped urgency, dropped red flags),
  and the target is the list of unsupported claims, each with severity + an exact transcript quote.
  Evaluated as recall on corrupted rows and false-positive rate on clean rows.

## Training (Kaggle)

1. `kaggle/train_sft.py` — QLoRA (r=32, α=64, 4-bit) on **Qwen3-4B**, 1 epoch, lr 2e-4, cosine,
   packing OFF (JSON correctness needs full attention), ChatML format. Measured ~12h/epoch on 1 T4,
   ~6h on T4 x2 via `torchrun --standalone --nproc_per_node=2` (3 epochs ~36h exceed Kaggle's ~9h
   session cap; no resume support).
   Alternative base: Phi-4-mini (3.8B, MIT) — switch via config.
2. `kaggle/dpo_stage2.py` — DPO on split D with asymmetric safety penalty (CoRFu-style):
   hallucinated-fact responses punished harder than minor errors.
3. `kaggle/export_gguf.py` — merge adapters → GGUF Q4_K_M (~2.6GB) via Unsloth.

## Evaluation (defines "good")

- **Extraction:** schema-valid JSON rate, field-level fuzzy F1 on held-out generated set (2k),
  meds/allergies recall (safety-critical), urgency agreement.
- **Grounded QA:** claim-support rate (answer claims must appear in cited context),
  abstention correctness, hallucination-trap pass rate (MHB-style).
- **Safety:** refusal rate on out-of-scope prompts, emergency escalation recall on red-flag cases.
- **Knowledge sanity (bonus):** MedQA 4-option subset (know its integrity flaws).
- **Device:** latency (<2s target), RAM, cold-start on slowest target phone.

## Deployment

- llama.cpp/MLC on phone, `.gbnf` grammar (`configs/extraction.gbnf`) for JSON decoding.
- On-device KB: compact vector store or BM25 over Mayo/WebMD chunks.
- Cloud fallback only for differentials/interactions; only de-identified JSON leaves the device.

## Milestones

1. [x] Generators + seed data (this repo)
2. [x] Dataset assembled locally: 37.8k train / 3.8k val (extraction 22k, guardrail 16.5k, grounded QA 2k, safety 1k)
3. [ ] Upload to Kaggle dataset → SFT Qwen3-4B → eval extraction F1 ≥ 0.90, schema-valid ≥ 99%
4. [ ] DPO stage → hallucination rate < 5% on trap set, abstain-correct ≥ 95%
5. [ ] Guardrail eval: corrupt-recall ≥ 0.95, clean-specificity ≥ 0.90, evidence-rate ≥ 0.85
6. [ ] GGUF export → on-phone latency test
7. [ ] Physician review of 300 outputs before any real-world use
