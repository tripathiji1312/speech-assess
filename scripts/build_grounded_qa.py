"""Build grounded-QA training data from your Mayo/WebMD-style corpus.

Your existing jsonl files (symptoms.jsonl, drugs.jsonl) become the *evidence KB*.
Each (question, answer, source) triple is converted into:

  SFT examples:   (context chunk + question) -> grounded answer with citation
                  + abstain examples when the question is paired with an
                  unrelated chunk (correct behavior: "not covered")
  DPO examples:   (context + question) with chosen = grounded answer,
                  rejected = answer from a different topic (hallucination trap)

Usage:
    python scripts/build_grounded_qa.py data/raw/symptoms.jsonl data/raw/drugs.jsonl \
        -o dataset/grounded_qa.jsonl --dpo -o-dpo dataset/dpo.jsonl

Input formats accepted (any mix of files):
  1. {"_id": ..., "conversations": [{"from":"human","value":...},{"from":"gpt","value":...}], "source": "Mayo Clinic"}
  2. {"text": "...", "source": "..."}   raw text gets chunked
"""

import argparse
import json
import random
import re
from pathlib import Path

ABSTAIN_ANSWER = "Answer: not_covered\nSource: none"
COVERED_TEMPLATE = "Answer: {text}\nSource: {source}"

SYSTEM_PROMPT = (
    "You are a health information assistant. Answer the question using ONLY the "
    "provided context. Quote or closely paraphrase the context; do not add outside "
    "knowledge, numbers, or doses that are not in the context. Always end with the "
    'line "Source: <source name>". If the context does not answer the question, '
    "respond exactly: 'Answer: not_covered\\nSource: none'."
)

TOPIC_RE = re.compile(r"^\s*(?:The\s+)?([A-Z][\w\-' ]{1,40}?)\s+(?:is|are|refers to|may be|can be|include|includes|works|helps)\b", re.M)
QUESTION_TOPIC_RE = [
    re.compile(r"what\s+is\s+(?:a|an|the)?\s*([\w\-' ]{1,40}?)\??$", re.I),
    re.compile(r"what\s+are\s+the?\s*([\w\-' ]{1,40}?)\??$", re.I),
    re.compile(r"how\s+does\s+([\w\-' ]{1,40}?)\s+work", re.I),
    re.compile(r"who\s+should\s+not\s+use\s+([\w\-' ]{1,40}?)\??$", re.I),
    re.compile(r"what\s+is\s+([\w\-' ]{1,40}?)\s+used\s+for", re.I),
]


def extract_topic(text, question=None):
    m = TOPIC_RE.search(text)
    if m:
        return m.group(1).strip().strip(":,;")
    if question:
        for rx in QUESTION_TOPIC_RE:
            m = rx.search(question)
            if m:
                t = m.group(1).strip().strip("?").strip()
                if t and len(t) > 2:
                    return t
    return None


def q_variants(original, topic, rng):
    vs = [original] if original.strip() else []
    if topic:
        t = topic
        if t and t[0].isupper() and len(t) > 1 and t[1].islower():
            t = t[0].lower() + t[1:]
        vs += [f"Tell me about {t}.",
               f"What can you tell me about {t}?",
               f"What should I know about {t}?"]
    return vs

def read_corpus(paths):
    entries = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"WARN: {p} not found, skipping")
            continue
        if p.suffix == ".jsonl":
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "conversations" in obj:
                        human = next((m["value"] for m in obj["conversations"] if m.get("from") == "human"), "")
                        gpt = next((m["value"] for m in obj["conversations"] if m.get("from") == "gpt"), "")
                        entries.append({"q": human.strip(), "a": gpt.strip(), "src": obj.get("source", "unknown")})
                    elif "text" in obj:
                        entries.append({"q": "", "a": obj["text"].strip(), "src": obj.get("source", "unknown")})
        else:
            text = p.read_text()
            chunks = [c.strip() for c in re.split(r"\n{2,}", text) if len(c.strip()) > 80]
            entries += [{"q": "", "a": c, "src": p.stem} for c in chunks]
    return [e for e in entries if e["a"]]


def chunk_answer(text, max_len=700):
    if len(text) <= max_len:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > max_len and cur:
            out.append(cur.strip())
            cur = s
        else:
            cur += " " + s
    if cur.strip():
        out.append(cur.strip())
    return out[0]


def to_messages(system, user, gpt):
    return [
        {"from": "system", "value": system},
        {"from": "human", "value": user},
        {"from": "gpt", "value": gpt},
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="jsonl / txt / md files of the corpus")
    ap.add_argument("-o", "--out", default="dataset/grounded_qa.jsonl")
    ap.add_argument("--o-dpo", default=None, help="write DPO pairs to this file (JSONL)")
    ap.add_argument("--abstain-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    corpus = read_corpus(args.inputs)
    if not corpus:
        print("ERROR: empty corpus. Check input paths.")
        raise SystemExit(1)
    print(f"corpus: {len(corpus)} entries")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_q = set()
    seen_dpo = set()
    sft_rows, dpo_rows = [], []
    for i, e in enumerate(corpus):
        chunk = chunk_answer(e["a"])
        topic = extract_topic(chunk, e["q"])
        for vi, q in enumerate(q_variants(e["q"], topic, rng)):
            key = re.sub(r"\s+", " ", q.lower())
            if key in seen_q:
                continue
            seen_q.add(key)
            user = f"Context:\n{chunk}\n\nQuestion: {q}"
            if rng.random() < args.abstain_ratio:
                other = corpus[rng.randrange(len(corpus))]
                while other is e:
                    other = corpus[rng.randrange(len(corpus))]
                user = f"Context:\n{chunk_answer(other['a'])}\n\nQuestion: {q}"
                sft_rows.append({
                    "_id": f"gqa-abstain-{i:06d}-{vi:02d}",
                    "source": e["src"],
                    "conversations": to_messages(SYSTEM_PROMPT, user, ABSTAIN_ANSWER),
                })
            else:
                gpt = COVERED_TEMPLATE.format(text=chunk, source=e["src"])
                sft_rows.append({
                    "_id": f"gqa-{i:06d}-{vi:02d}",
                    "source": e["src"],
                    "conversations": to_messages(SYSTEM_PROMPT, user, gpt),
                })
            if args.o_dpo and rng.random() < 0.5:
                other = corpus[rng.randrange(len(corpus))]
                while other is e:
                    other = corpus[rng.randrange(len(corpus))]
                for _ in range(10):
                    o_topic = extract_topic(chunk_answer(other["a"]), other["q"])
                    if o_topic is None or (topic and o_topic.lower() != topic.lower()):
                        break
                    other = corpus[rng.randrange(len(corpus))]
                    while other is e:
                        other = corpus[rng.randrange(len(corpus))]
                dpo_key = (user, chunk, other["src"])
                if dpo_key not in seen_dpo:
                    seen_dpo.add(dpo_key)
                    dpo_rows.append({
                        "_id": f"dpo-{i:06d}-{vi:02d}",
                        "source": e["src"],
                        "system": SYSTEM_PROMPT,
                        "prompt": user,
                        "chosen": COVERED_TEMPLATE.format(text=chunk, source=e["src"]),
                        "rejected": COVERED_TEMPLATE.format(text=chunk_answer(other["a"]), source=other["src"]),
                    })

    with open(out_path, "w") as f:
        for r in sft_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.o_dpo:
        with open(args.o_dpo, "w") as f:
            for r in dpo_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(sft_rows)} SFT rows -> {out_path}")
        print(f"wrote {len(dpo_rows)} DPO pairs -> {args.o_dpo}")
    else:
        print(f"wrote {len(sft_rows)} SFT rows -> {out_path}")


if __name__ == "__main__":
    main()
