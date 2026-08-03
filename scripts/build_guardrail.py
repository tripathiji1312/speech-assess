"""Generate claim-support guardrail data: verify an extraction draft against a transcript.

Implements the "confabulation elimination" stage from Abridge's whitepaper +
evidence-linked verification from the ACL 2021 modular-summarization line.

For each synthetic case we render the transcript, then with probability
--corrupt-ratio we corrupt EXACTLY ONE field of the gold extraction JSON
(invented/ swapped / dropped / contradicted). The training target is a
verification JSON listing every unsupported claim with:
    field, claim, problem, severity (low|medium|high), evidence (transcript quote)

Since we know the true case and the corruption we injected, labels are exact.

Usage:
    python scripts/build_guardrail.py --n-train 15000 --n-val 1500 -o dataset/
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcript_simulator import (  # noqa: E402
    COMPLAINTS, MEDS, Case, render_gold, render_transcript,
)

FILLER_RE = re.compile(
    r"^(?:uh|um|like|you know|hmm|i mean|so|actually|ji|sahab|bahut|thoda|"
    r"kabhi kabhi|bilkul|\s|,)+", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a clinical claim verifier. Given a chat transcript and an "
    "extraction JSON draft, find every claim in the draft that is NOT fully "
    "supported by the transcript. Output strict JSON: "
    '{"issues": [{"field": ..., "claim": ..., "problem": ..., '
    '"severity": "low"|"medium"|"high", "evidence": ...}], "summary": ...}. '
    "problem is one of: not in transcript, contradicts transcript, missing from draft. "
    'evidence must be an exact quote from the transcript. If every claim is '
    'supported, output {"issues": [], "summary": "ok"}. No text outside the JSON.'
)

SEVERITY = {
    "med_invent": "high",
    "med_swap": "high",
    "med_drop": "high",
    "med_dose": "medium",
    "redflag_invent": "high",
    "redflag_drop": "high",
    "urgency_flip": "high",
    "allergy_invent": "high",
    "vital_invent": "medium",
    "vital_drop": "low",
    "hpi_invent": "medium",
    "severity_flip": "medium",
}

INVENTED_SYMPTOMS = [
    "Reports intermittent dizziness in the mornings.",
    "Reports night sweats over the past week.",
    "Reports a new rash on the left forearm.",
    "Reports ringing in the ears since last night.",
    "Reports joint stiffness in both knees.",
]

INVENTED_RED_FLAGS = [
    "blood in the stool",
    "vomiting blood",
    "chest pain",
    "shortness of breath",
    "fainting",
    "numbness in the legs",
    "vision changes",
]


def find_line(transcript, needles):
    for line in transcript.split("\n"):
        for n in needles:
            if n and n.lower() in line.lower():
                return line
    return ""


def meds_line(transcript):
    return find_line(transcript, ["I take", "nothing right now"])


def vitals_line(transcript):
    return find_line(transcript, ["temperature", "blood pressure", "heart rate", "oxygen"])


def redflag_line(transcript, flag_text):
    words = flag_text.split()[:4]
    return find_line(transcript, [" ".join(words)])


def redflag_contradiction_line(transcript, flag_text):
    """A negated probe line ('Patient: no chest pain.') that contradicts an
    invented red flag."""
    words = flag_text.split()[:4]
    for line in transcript.split("\n"):
        low = line.lower()
        if low.startswith("patient:") and " no " in low:
            if all(w in low for w in words):
                return line
    return ""


def negated_probe_flags(transcript):
    """Red flags the patient explicitly denied ('Patient: no, uh, fever.').
    Inventing one of these gives a guaranteed contradiction line as evidence."""
    flags = []
    for line in transcript.split("\n"):
        low = line.lower()
        m = re.match(r"^patient:\s*no\s+(.*?)\s*\.?$", low)
        if not m:
            continue
        f = FILLER_RE.sub("", m.group(1)).strip().strip(" ,.")
        if f and len(f.split()) <= 6:
            flags.append(f)
    return flags


FIELD_QUESTION_NEEDLES = {
    "medications": ["taking any medications"],
    "vitals": ["checked your vitals"],
    "allergies": ["allergies"],
    "red_flags": [],
    "hpi": [],
    "urgency": [],
    "severity_hpi": [],
}


def build_issue(case, transcript, field, claim, problem, corruption):
    if corruption in ("med_drop", "redflag_drop", "vital_drop"):
        if field == "medications":
            ev = meds_line(transcript)
        elif field == "red_flags":
            ev = redflag_line(transcript, claim)
        else:
            ev = vitals_line(transcript)
    elif field == "urgency":
        ev = ""
        for flag, _ in case.red_present:
            ev = redflag_line(transcript, flag)
            if ev:
                break
        if not ev:
            ev = vitals_line(transcript) or next(
                (l for l in transcript.split("\n") if l.startswith("Patient:")), "")
    elif field == "severity_hpi":
        ev = find_line(transcript, ["It was"])
    elif field == "medications":
        ev = meds_line(transcript)
    elif field == "allergies":
        ev = find_line(transcript, ["allerg"])
    elif field == "vitals":
        ev = vitals_line(transcript)
    elif field == "red_flags":
        ev = redflag_contradiction_line(transcript, claim)
    elif field == "hpi":
        ev = next((l for l in transcript.split("\n") if l.startswith("Patient:")), "")
    else:
        ev = ""
    if not ev and field in FIELD_QUESTION_NEEDLES:
        ev = find_line(transcript, FIELD_QUESTION_NEEDLES[field])
    return {
        "field": field,
        "claim": claim,
        "problem": problem,
        "severity": SEVERITY[corruption],
        "evidence": ev,
    }


def corrupt_gold(case, rng):
    """Return (corrupted_json, issue). Exactly one field is corrupted."""
    gold = render_gold(case)
    transcript = render_transcript(case)
    kinds = list(SEVERITY)
    rng.shuffle(kinds)

    for kind in kinds:
        if kind == "med_invent":
            others = [m for m in MEDS if m[0] not in [x["name"] for x in case.meds]]
            if not others:
                continue
            name, dose, freq, _, _ = rng.choice(others)
            entry = {"name": name, "dose": dose, "frequency": freq}
            gold["medications"].append(entry)
            return gold, build_issue(case, transcript, "medications",
                                     f"{name} {dose}, {freq}", "not in transcript", kind)

        if kind == "med_swap" and len(case.meds) >= 1:
            others = [m for m in MEDS if m[0] not in [x["name"] for x in case.meds]]
            if not others:
                continue
            name, dose, freq, _, _ = rng.choice(others)
            entry = {"name": name, "dose": dose, "frequency": freq}
            gold["medications"][rng.randrange(len(gold["medications"]))] = entry
            return gold, build_issue(case, transcript, "medications",
                                     f"{name} {dose}, {freq}", "not in transcript", kind)

        if kind == "med_drop" and len(case.meds) >= 1:
            dropped = gold["medications"].pop(rng.randrange(len(gold["medications"])))
            claim = f"{dropped['name']} {dropped['dose']}, {dropped['frequency']}"
            return gold, build_issue(case, transcript, "medications",
                                     claim, "missing from draft", kind)

        if kind == "med_dose" and len(case.meds) >= 1:
            m = rng.choice(gold["medications"])
            real = next(x for x in case.meds if x["name"] == m["name"])
            if rng.random() < 0.5:
                try:
                    m["dose"] = str(float(real["dose"].split()[0]) * 2).rstrip("0").rstrip(".") + " " + " ".join(real["dose"].split()[1:])
                except ValueError:
                    m["dose"] = "500 mg"
            else:
                m["frequency"] = "every 12 hours" if "once daily" in real["frequency"] else "once daily"
            claim = f"{m['name']} {m['dose']}, {m['frequency']}"
            return gold, build_issue(case, transcript, "medications",
                                     claim, "contradicts transcript", kind)

        if kind == "redflag_invent":
            denied = negated_probe_flags(transcript)
            if denied:
                flag = rng.choice(denied)
                gold["red_flags"].append(flag)
                ev = redflag_contradiction_line(transcript, flag)
                issue = build_issue(case, transcript, "red_flags",
                                    flag, "contradicts transcript", kind)
                issue["evidence"] = ev
                return gold, issue
            available = [
                f for f in INVENTED_RED_FLAGS if f not in transcript.lower()
            ]
            if not available:
                continue
            flag = rng.choice(available)
            gold["red_flags"].append(flag)
            return gold, build_issue(case, transcript, "red_flags",
                                     flag, "not in transcript", kind)

        if kind == "redflag_drop" and case.red_present:
            dropped = gold["red_flags"].pop(rng.randrange(len(gold["red_flags"])))
            return gold, build_issue(case, transcript, "red_flags",
                                     dropped, "missing from draft", kind)

        if kind == "urgency_flip":
            others = [u for u in ["routine", "urgent", "emergency"] if u != gold["urgency"]]
            gold["urgency"] = rng.choice(others)
            gold["escalate"] = gold["urgency"] != "routine"
            claim = f'urgency "{gold["urgency"]}"'
            return gold, build_issue(case, transcript, "urgency",
                                     claim, "contradicts transcript", kind)

        if kind == "allergy_invent":
            pool = ["Penicillin", "Sulfa drugs", "Latex", "Peanuts", "Ibuprofen"]
            avail = [a for a in pool if a not in [x.lower() for x in gold["allergies"]]]
            if not avail:
                continue
            al = rng.choice(avail)
            gold["allergies"].append(al)
            return gold, build_issue(case, transcript, "allergies",
                                     al, "not in transcript", kind)

        if kind == "vital_invent":
            null_keys = [k for k, v in gold["vitals"].items() if v is None]
            if not null_keys:
                continue
            k = rng.choice(null_keys)
            vals = {"temp": "104.2F", "bp": "170/105", "hr": 132, "spo2": 87}
            gold["vitals"][k] = vals[k]
            return gold, build_issue(case, transcript, "vitals",
                                     f"{k} {vals[k]}", "not in transcript", kind)

        if kind == "vital_drop":
            present = [k for k, v in gold["vitals"].items() if v is not None]
            if not present:
                continue
            k = rng.choice(present)
            old = gold["vitals"][k]
            gold["vitals"][k] = None
            return gold, build_issue(case, transcript, "vitals",
                                     f"{k} {old}", "missing from draft", kind)

        if kind == "hpi_invent":
            avail = [s for s in INVENTED_SYMPTOMS
                     if s.split()[1] not in transcript.lower()]
            if not avail:
                continue
            s = rng.choice(avail)
            gold["hpi"] += " " + s
            return gold, build_issue(case, transcript, "hpi",
                                     s, "not in transcript", kind)

        if kind == "severity_flip" and "Severity described" in gold["hpi"]:
            new = "severe" if case.severity != "severe" else "mild"
            gold["hpi"] = gold["hpi"].replace("Severity described as " + case.severity + ".",
                                              f"Severity described as {new}.")
            claim = f"Severity described as {new}."
            return gold, build_issue(case, transcript, "hpi",
                                     claim, "contradicts transcript", kind)

    return None, None


def render_verdict(case, transcript, gold, issue):
    if issue is None:
        return {"issues": [], "summary": "ok"}
    return {"issues": [issue], "summary": "issues_found"}


def generate(n, seed, corrupt_ratio, start_id=0):
    rng = random.Random(seed)
    out = []
    keys = list(COMPLAINTS)
    for i in range(n):
        key = keys[i % len(keys)] if i < len(keys) else rng.choice(keys)
        case = Case(rng, key, noise=2, lang="en")
        transcript = render_transcript(case)
        gold, issue = corrupt_gold(case, rng)
        if gold is None:
            issue = None
            gold = render_gold(case)
        elif rng.random() > corrupt_ratio:
            issue = None
            gold = render_gold(case)
        verdict = render_verdict(case, transcript, gold, issue)
        out.append({
            "_id": f"guard-{start_id + i:06d}",
            "corruption": None if issue is None else issue["field"],
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": "Transcript:\n" + transcript
                                           + "\n\nDraft extraction:\n"
                                           + json.dumps(gold, ensure_ascii=False, separators=(",", ":"))},
                {"from": "gpt", "value": json.dumps(verdict, ensure_ascii=False, separators=(",", ":"))},
            ],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--corrupt-ratio", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-o", "--out", default="dataset")
    args = ap.parse_args()

    n_train, n_val = (500, 100) if args.quick else (args.n_train, args.n_val)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = generate(n_train, args.seed, args.corrupt_ratio, 0)
    val = generate(n_val, args.seed + 1, args.corrupt_ratio, n_train)

    with open(out / "guardrail_train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out / "guardrail_val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    from collections import Counter
    corrupt = Counter(ex["corruption"] for ex in train if ex["corruption"])
    print(f"wrote {n_train} train + {n_val} val -> {out}/")
    print(f"corrupt rate: {sum(1 for e in train if e['corruption']) / len(train):.2%}")
    print("corruption type mix:", dict(corrupt))


if __name__ == "__main__":
    main()
