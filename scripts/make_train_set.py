"""Assemble all generated data into final ChatML train/val splits.

Inputs (any subset that exists):
    dataset/extraction_train.jsonl   (already split)
    dataset/extraction_val.jsonl
    dataset/grounded_qa.jsonl        (split here, 90/10)
    dataset/safety.jsonl             (split here, 90/10)
    extra: any additional jsonl files in dataset/extra/ are merged 90/10

Outputs (into dataset/final/):
    train.jsonl, val.jsonl           final ChatML rows
    stats.json                       per-task counts + split sizes

Usage:
    python scripts/make_train_set.py -o dataset/final/
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def content_hash(obj):
    h = hashlib.sha256()
    for m in obj["conversations"]:
        h.update(m["value"].encode("utf-8", errors="ignore"))
    return h.hexdigest()


def assign_split(ex, rng):
    return rng.random() < 0.9


def load(path, require=True):
    if not path.exists():
        if require:
            raise SystemExit(f"ERROR: {path} not found")
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="dataset")
    ap.add_argument("-o", "--out", default="dataset/final")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    tasks = {
        "extraction": [
            (root / "extraction_train.jsonl", False),
            (root / "extraction_val.jsonl", False),
        ],
        "guardrail": [
            (root / "guardrail_train.jsonl", False),
            (root / "guardrail_val.jsonl", False),
        ],
        "grounded_qa": [(root / "grounded_qa.jsonl", True)],
        "safety": [(root / "safety.jsonl", True)],
    }
    extra_dir = root / "extra"
    if extra_dir.exists():
        tasks["extra"] = [(p, True) for p in sorted(extra_dir.glob("*.jsonl"))]

    train, val = [], []
    seen = set()
    stats = {}
    for task, files in tasks.items():
        stats[task] = {"rows": 0, "train": 0, "val": 0}
        for path, do_split in files:
            for ex in load(path, require=path.exists()):
                h = content_hash(ex)
                if h in seen:
                    continue
                seen.add(h)
                if ex.get("_id"):
                    seen.add(ex["_id"])
                if do_split:
                    bucket = val if not assign_split(ex, rng) else train
                elif "val" in path.name:
                    bucket = val
                else:
                    bucket = train
                bucket.append(ex)
                stats[task]["train" if bucket is train else "val"] += 1
                stats[task]["rows"] += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(train):
        ex["_id"] = f"train-{i:06d}"
    for i, ex in enumerate(val):
        ex["_id"] = f"val-{i:06d}"

    def slim(ex):
        return {"_id": ex["_id"], "conversations": ex["conversations"]}

    with open(out / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(slim(ex), ensure_ascii=False) + "\n")
    with open(out / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(slim(ex), ensure_ascii=False) + "\n")

    def nchars(rows):
        return sum(len(m["value"]) for r in rows for m in r["conversations"])

    stats["_summary"] = {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_est_tokens": int(nchars(train) / 4),
        "val_est_tokens": int(nchars(val) / 4),
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"train: {len(train)} rows, val: {len(val)} rows -> {out}/")
    for task in tasks:
        print(f"  {task}: {stats[task]}")
    print("summary:", stats["_summary"])


if __name__ == "__main__":
    main()
