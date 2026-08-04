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
| 2026-08-04 | findings-2026-08-04.md | 3rd smoke test still crashed: transformers 5.5.0 Qwen3DecoderLayer inherits GradientCheckpointingLayer whose __call__ uses a CAPTURED `_gradient_checkpointing_func` partial (bound reentrant by unsloth's `prepare_model_for_kbit_training(..., use_reentrant=True)`), bypassing module-attribute wrappers. Fix: also overwrite `_gradient_checkpointing_func` per-module with non-reentrant partial (`use_reentrant=False, determinism_check="none"`), applied after get_peft_model + before trainer.train(); mock-tested PASS, py_compile PASS |
| 2026-08-04 | findings-2026-08-04.md | 4th smoke test crashed identically — but on STALE code: traceback line refs (306/314) match pre-fix file (now 344/351) and `[DDP fix]` print absent; fix was never pushed. Committed a6ebc6e + pushed to origin/main; mechanism 2 still UNTESTED on Kaggle — next test must show "forced non-reentrant checkpointing on 37 module(s)" |
| 2026-08-04 | findings-2026-08-04.md | 5th smoke test: DDP FIX VERIFIED — 37 modules patched, 3 steps completed (loss 2.628/3.055/2.829, train_loss 2.837), no "ready twice" error. New unrelated failure: save_pretrained_merged raced on both ranks (FileNotFoundError on shards, saving_utils.py:717). Fix: rank-guard merge to LOCAL_RANK==0; py_compile PASS; merge fix UNVERIFIED until next smoke test |
| 2026-08-04 | findings-2026-08-04.md | 6th smoke test FULLY GREEN: DDP fix + merge rank-guard verified end-to-end — 3 steps trained, merge 100% ("Merge process complete. Saved to /kaggle/working/sft_test"), clean exit. Measured ~8-9.5 s/step → full run ≈ 2616 steps ≈ 6h, within session cap; VRAM at batch 4/seq 2048 empirically OK on 2xT4. READY FOR FULL ~6H RUN |