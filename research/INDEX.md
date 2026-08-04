# Research Log Index

Every planner run writes its full deep-researcher findings here. Read the
findings files listed below BEFORE starting new work — they contain verified
project context that the planner already gathered, so you can reuse it instead
of re-searching.

## How entries get here

- The planner appends every subagent report, verbatim, to `research/findings-<YYYY-MM-DD>.md`.
- The planner keeps this index up to date at the end of each run.

## Findings

| Date | File | Topic studied |
| ---- | ---- | ------------- |
| 2026-08-04 | findings-2026-08-04.md | DDP "marked as ready twice" crash in Unsloth 2xT4 QLoRA SFT: root cause (reentrant smart GC vs DDP), unsloth LLM-path gap (PR #3751 is VLM-only), fix = native non-reentrant GC (`use_gradient_checkpointing=True`), 3-step Kaggle verification strategy, codebase config audit, VRAM item UNVERIFIED with batch-2 fallback |