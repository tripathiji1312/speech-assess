"""Generate synthetic clinical chat transcripts -> structured extraction JSON.

No API / no model required. The simulator knows the ground truth it injects,
so the gold labels are exactly correct by construction.

Usage:
    python scripts/transcript_simulator.py --n-train 20000 --n-val 2000 -o dataset/
    python scripts/transcript_simulator.py --quick   # small smoke-test run

Output (into OUT dir):
    extraction_train.jsonl   list of {"_id", "noise", "conversations": [system, human, gpt]}
    extraction_val.jsonl     same, held out
    schema.json              the extraction schema (used by eval + GBNF grammar)
    topic_map.json           complaint/med -> KB topics (for on-device retrieval routing)
"""

import argparse
import json
import random
from pathlib import Path

DEFAULT_SEED = 42
PHI_TOKEN = "[NAME]"

VITALS_SCHEMA = ["temp", "bp", "hr", "spo2"]

SCHEMA = {
    "chief_complaint": "string - short phrase, only what patient said",
    "hpi": "string - 2-4 sentence narrative of the present illness from the transcript only",
    "vitals": {"temp": "string|null", "bp": "string|null", "hr": "int|null", "spo2": "int|null"},
    "medications": ["{name, dose, frequency}"],
    "allergies": ["string"],
    "red_flags": ["string"],
    "missing_info": ["string"],
    "urgency": "enum(emergency, urgent, routine)",
    "escalate": "bool - true iff urgency != routine",
}

COMPLAINTS = {
    "abdominal_pain": {
        "name": "abdominal pain",
        "phrases": ["my stomach hurts", "I have pain in my belly", "abdominal pain",
                    "my stomach has been hurting", "pain in my abdomen", "stomach ache"],
        "duration": ["3 days", "about a week", "since yesterday", "2 weeks", "the last few days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["nausea", "vomiting", "bloating", "diarrhea", "loss of appetite", "heartburn"],
        "red_flags": [
            ("blood in the stool", "urgent"),
            ("vomiting blood", "emergency"),
            ("pain so severe I can't move", "emergency"),
        ],
        "denials": ["no nausea", "no vomiting", "no fever", "no bleeding"],
        "base_urgency": "routine",
        "topics": ["Abdominal pain", "Gastroenteritis", "IBS", "Anal pain"],
    },
    "headache": {
        "name": "headache",
        "phrases": ["I have a headache", "my head hurts", "bad headaches", "migraine"],
        "duration": ["2 days", "since this morning", "almost a week", "4 days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["light sensitivity", "neck stiffness", "blurred vision", "nausea"],
        "red_flags": [
            ("worst headache of my life", "emergency"),
            ("headache with stiff neck and fever", "emergency"),
            ("sudden vision changes", "urgent"),
        ],
        "denials": ["no vision changes", "no neck stiffness", "no numbness"],
        "base_urgency": "routine",
        "topics": ["Headache", "Migraine", "Tension headache"],
    },
    "chest_pain": {
        "name": "chest pain",
        "phrases": ["pain in my chest", "my chest hurts", "chest tightness"],
        "duration": ["2 hours", "since this morning", "1 day", "a few days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["shortness of breath", "sweating", "nausea", "dizziness"],
        "red_flags": [
            ("pain spreading to my arm and jaw", "emergency"),
            ("crushing chest pain with sweating", "emergency"),
            ("chest pain with shortness of breath at rest", "emergency"),
        ],
        "denials": ["no shortness of breath", "no sweating", "no pain in the arm"],
        "base_urgency": "urgent",
        "topics": ["Chest pain", "Angina", "Heart attack warning signs", "Acid reflux"],
    },
    "shortness_of_breath": {
        "name": "shortness of breath",
        "phrases": ["I'm short of breath", "hard to breathe", "breathing trouble", "I can't catch my breath"],
        "duration": ["2 days", "since last night", "a week", "1 day"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["wheezing", "cough", "chest tightness", "dizziness", "swelling in the legs"],
        "red_flags": [
            ("shortness of breath while sitting still", "emergency"),
            ("lips turning blue", "emergency"),
            ("fainting", "emergency"),
        ],
        "denials": ["no chest pain", "no wheezing", "no swelling"],
        "base_urgency": "urgent",
        "topics": ["Shortness of breath", "Asthma", "COPD", "Pneumonia"],
    },
    "fever": {
        "name": "fever",
        "phrases": ["I have a fever", "temperature is up", "running a fever"],
        "duration": ["2 days", "3 days", "since yesterday", "about a week"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["chills", "body aches", "sweating", "headache", "fatigue"],
        "red_flags": [
            ("fever with a stiff neck", "emergency"),
            ("fever over 104 with confusion", "emergency"),
            ("fever in a baby under 3 months", "urgent"),
        ],
        "denials": ["no rash", "no confusion", "no stiff neck"],
        "base_urgency": "routine",
        "topics": ["Fever", "Flu", "Infection", "Elevated liver enzymes"],
    },
    "cough": {
        "name": "cough",
        "phrases": ["I have a bad cough", "coughing a lot", "dry cough", "cough with phlegm"],
        "duration": ["5 days", "2 weeks", "since last month", "10 days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["fever", "phlegm", "wheezing", "chest pain", "runny nose"],
        "red_flags": [
            ("coughing up blood", "urgent"),
            ("cough with severe breathing difficulty", "emergency"),
        ],
        "denials": ["no fever", "no blood", "no wheezing"],
        "base_urgency": "routine",
        "topics": ["Cough", "Bronchitis", "Pneumonia", "Asthma"],
    },
    "dizziness": {
        "name": "dizziness",
        "phrases": ["I feel dizzy", "spinning sensation", "lightheaded", "feeling faint"],
        "duration": ["1 day", "3 days", "since last week", "2 weeks"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["nausea", "blurred vision", "weakness", "headache"],
        "red_flags": [
            ("fainted", "urgent"),
            ("dizziness with chest pain", "emergency"),
            ("dizziness with weakness on one side", "emergency"),
        ],
        "denials": ["no fainting", "no chest pain", "no weakness"],
        "base_urgency": "routine",
        "topics": ["Dizziness", "Vertigo", "Anemia"],
    },
    "back_pain": {
        "name": "back pain",
        "phrases": ["pain in my lower back", "my back hurts", "back pain"],
        "duration": ["1 week", "3 weeks", "since last month", "a few days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["leg pain", "numbness", "tingling", "stiffness"],
        "red_flags": [
            ("numbness around the groin", "emergency"),
            ("loss of bladder control", "emergency"),
            ("back pain with fever", "urgent"),
        ],
        "denials": ["no leg numbness", "no fever", "no weakness"],
        "base_urgency": "routine",
        "topics": ["Back pain", "Sciatica", "Muscle strain"],
    },
    "nausea_vomiting": {
        "name": "nausea and vomiting",
        "phrases": ["I keep throwing up", "nauseous all the time", "vomiting since yesterday"],
        "duration": ["1 day", "2 days", "3 days", "since last night"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["diarrhea", "stomach cramps", "dehydration", "fever"],
        "red_flags": [
            ("can't keep any fluids down for 24 hours", "urgent"),
            ("vomiting blood", "emergency"),
            ("signs of dehydration in a child", "urgent"),
        ],
        "denials": ["no blood", "no severe dehydration"],
        "base_urgency": "routine",
        "topics": ["Nausea and vomiting", "Gastroenteritis", "Dehydration"],
    },
    "rash": {
        "name": "rash",
        "phrases": ["I have a rash on my arm", "red spots on my skin", "itchy rash"],
        "duration": ["2 days", "1 week", "since last weekend", "a few days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["itching", "fever", "swelling", "pain"],
        "red_flags": [
            ("rash with difficulty breathing", "emergency"),
            ("rash with swelling of the face or lips", "emergency"),
            ("blistering rash", "urgent"),
        ],
        "denials": ["no fever", "no swelling", "no breathing trouble"],
        "base_urgency": "routine",
        "topics": ["Rash", "Allergic reaction", "Eczema"],
    },
    "sore_throat": {
        "name": "sore throat",
        "phrases": ["my throat hurts", "sore throat", "pain when swallowing"],
        "duration": ["3 days", "1 week", "since last week", "2 days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["fever", "swollen glands", "cough", "difficulty swallowing"],
        "red_flags": [
            ("trouble breathing", "emergency"),
            ("drooling and can't swallow", "emergency"),
            ("throat closing up", "emergency"),
        ],
        "denials": ["no fever", "no trouble breathing"],
        "base_urgency": "routine",
        "topics": ["Sore throat", "Strep throat", "Tonsillitis"],
    },
    "joint_pain": {
        "name": "joint pain",
        "phrases": ["pain in my knee", "joints hurt", "swollen knee", "pain in my wrist"],
        "duration": ["2 weeks", "1 month", "since last month", "a few days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["swelling", "stiffness in the morning", "redness", "fever"],
        "red_flags": [
            ("hot swollen joint with fever", "urgent"),
            ("joint pain after an injury with deformity", "urgent"),
        ],
        "denials": ["no fever", "no redness", "no recent injury"],
        "base_urgency": "routine",
        "topics": ["Joint pain", "Arthritis", "Gout"],
    },
    "uti": {
        "name": "urinary symptoms",
        "phrases": ["burning when I pee", "urinary burning", "frequent urination", "painful urination"],
        "duration": ["2 days", "1 week", "since last week", "3 days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["fever", "back pain", "blood in urine", "urgency"],
        "red_flags": [
            ("fever with back pain", "urgent"),
            ("severe flank pain", "urgent"),
        ],
        "denials": ["no fever", "no back pain", "no blood"],
        "base_urgency": "routine",
        "topics": ["Urinary tract infection", "UTI symptoms", "Kidney infection"],
    },
    "leg_swelling": {
        "name": "leg swelling",
        "phrases": ["my leg is swollen", "swelling in one leg", "my ankle is puffy"],
        "duration": ["1 day", "3 days", "since last week", "2 days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["pain in the leg", "redness", "warmth", "shortness of breath"],
        "red_flags": [
            ("one-sided leg swelling with pain", "urgent"),
            ("swelling with shortness of breath", "emergency"),
        ],
        "denials": ["no shortness of breath", "no redness"],
        "base_urgency": "routine",
        "topics": ["Leg swelling", "Deep vein thrombosis", "Edema"],
    },
    "palpitations": {
        "name": "heart palpitations",
        "phrases": ["my heart is racing", "heart palpitations", "my heart skips beats"],
        "duration": ["2 hours", "1 day", "since last week", "a few days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["dizziness", "chest pain", "shortness of breath", "sweating"],
        "red_flags": [
            ("palpitations with fainting", "emergency"),
            ("palpitations with chest pain", "emergency"),
        ],
        "denials": ["no fainting", "no chest pain"],
        "base_urgency": "urgent",
        "topics": ["Palpitations", "Arrhythmia", "Anxiety"],
    },
    "insomnia": {
        "name": "trouble sleeping",
        "phrases": ["I can't sleep", "trouble sleeping", "insomnia", "waking up at night"],
        "duration": ["2 weeks", "1 month", "since last month", "3 weeks"],
        "onset": ["gradual", "sudden"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["anxiety", "stress", "daytime fatigue", "snoring"],
        "red_flags": [
            ("chest pain at night", "urgent"),
            ("gasping during sleep", "urgent"),
        ],
        "denials": ["no snoring", "no gasping"],
        "base_urgency": "routine",
        "topics": ["Insomnia", "Sleep hygiene", "Anxiety"],
    },
    "ear_pain": {
        "name": "ear pain",
        "phrases": ["pain in my ear", "my ear hurts", "earache"],
        "duration": ["2 days", "1 week", "since last week", "3 days"],
        "onset": ["sudden", "gradual"],
        "severity": ["mild", "moderate", "severe"],
        "associated": ["fever", "hearing loss", "drainage", "ear ringing"],
        "red_flags": [
            ("drainage from the ear", "urgent"),
            ("hearing loss", "urgent"),
            ("severe pain behind the ear with swelling", "urgent"),
        ],
        "denials": ["no drainage", "no fever"],
        "base_urgency": "routine",
        "topics": ["Ear pain", "Ear infection", "Otitis media"],
    },
}

MEDS = [
    ("Lisinopril", "10 mg", "once daily", "blood pressure", ["HTN", "hypertension", "high blood pressure"]),
    ("Metformin", "500 mg", "twice daily", "diabetes", ["diabetes", "sugar"]),
    ("Ibuprofen", "400 mg", "every 6 hours as needed", "pain and inflammation", ["pain", "arthritis"]),
    ("Acetaminophen", "500 mg", "every 6 hours as needed", "fever and pain", ["fever", "pain"]),
    ("Amoxicillin", "500 mg", "three times daily", "bacterial infection", ["infection"]),
    ("Atorvastatin", "20 mg", "once daily", "cholesterol", ["cholesterol"]),
    ("Omeprazole", "20 mg", "once daily before breakfast", "acid reflux", ["acidity", "reflux", "heartburn"]),
    ("Albuterol inhaler", "2 puffs", "as needed", "asthma", ["asthma", "wheezing"]),
    ("Metoprolol", "25 mg", "twice daily", "blood pressure", ["HTN", "heart"]),
    ("Levothyroxine", "75 mcg", "once daily on empty stomach", "thyroid", ["thyroid"]),
    ("Amlodipine", "5 mg", "once daily", "blood pressure", ["HTN"]),
    ("Losartan", "50 mg", "once daily", "blood pressure", ["HTN"]),
    ("Gabapentin", "300 mg", "three times daily", "nerve pain", ["nerve pain", "neuropathy"]),
    ("Sertraline", "50 mg", "once daily", "depression and anxiety", ["anxiety", "depression"]),
    ("Prednisone", "20 mg", "once daily for 5 days", "inflammation", ["inflammation", "asthma"]),
    ("Azithromycin", "250 mg", "once daily for 5 days", "infection", ["infection"]),
    ("Cetirizine", "10 mg", "once daily", "allergies", ["allergy", "itching"]),
    ("Loratadine", "10 mg", "once daily", "allergies", ["allergy"]),
    ("Ondansetron", "4 mg", "every 8 hours as needed", "nausea", ["nausea", "vomiting"]),
    ("Furosemide", "40 mg", "once daily", "fluid retention", ["swelling", "edema"]),
    ("Hydrochlorothiazide", "25 mg", "once daily", "blood pressure", ["HTN"]),
    ("Rosuvastatin", "10 mg", "once daily", "cholesterol", ["cholesterol"]),
    ("Duloxetine", "30 mg", "once daily", "nerve pain and depression", ["pain", "depression"]),
    ("Pregabalin", "75 mg", "twice daily", "nerve pain", ["nerve pain"]),
    ("Sumatriptan", "50 mg", "at onset, may repeat in 2 hours", "migraine", ["migraine", "headache"]),
    ("Fluticasone nasal spray", "2 sprays", "once daily", "allergic rhinitis", ["allergy", "stuffy nose"]),
    ("Montelukast", "10 mg", "once daily at night", "asthma", ["asthma"]),
    ("Doxycycline", "100 mg", "twice daily", "infection", ["infection"]),
    ("Naproxen", "220 mg", "twice daily as needed", "pain", ["pain", "arthritis"]),
    ("Meloxicam", "7.5 mg", "once daily", "joint pain", ["joint pain", "arthritis"]),
    ("Warfarin", "2 mg", "once daily", "blood thinner", ["clot", "blood thinner"]),
    ("Clopidogrel", "75 mg", "once daily", "blood thinner", ["clot"]),
    ("Insulin glargine", "10 units", "once daily at bedtime", "diabetes", ["diabetes", "sugar"]),
    ("Salmeterol inhaler", "1 puff", "twice daily", "asthma", ["asthma"]),
    ("Calcium carbonate", "500 mg", "twice daily as needed", "heartburn", ["acidity", "heartburn"]),
]

ALLERGIES = ["Penicillin", "Sulfa drugs", "Latex", "Peanuts", "Ibuprofen", "None known"]

HISTORY = ["diabetes", "high blood pressure", "asthma", "high cholesterol", "thyroid disease",
           "gastritis", "migraines", "no significant history"]

ABBREV = {
    "shortness of breath": "SOB",
    "high blood pressure": "HTN",
    "diabetes": "DM",
    "as needed": "PRN",
    "twice daily": "BID",
    "three times daily": "TID",
    "once daily": "QD",
    "emergency room": "ER",
    "urinary tract infection": "UTI",
}

FILLERS = ["uh", "um", "like", "you know", "hmm", "I mean", "so", "actually"]
DISFLUENCIES = ["I, I", "it, it started", "my, my stomach", "we, we went", "he, he said", "they, they told me"]

OPENERS = [
    "What brings you in today?",
    "What can I help you with today?",
    "Hello, what seems to be the problem?",
    "How can I help you?",
    "What's going on today?",
    "Tell me what's been bothering you.",
]

HINDI_FLAVOR = ["ji", "sahab", "bahut", "thoda", "kabhi kabhi", "bilkul", "chaliye dekhte hain"]


class Case:
    def __init__(self, rng, complaint_key, noise, lang):
        c = COMPLAINTS[complaint_key]
        self.rng = rng
        self.noise = noise
        self.lang = lang
        self.qid = f"gen-{rng.randint(0, 10**9):09d}"
        self.complaint_key = complaint_key
        self.phrase = rng.choice(c["phrases"])
        self.duration = rng.choice(c["duration"])
        self.onset = rng.choice(c["onset"])
        self.severity = rng.choice(c["severity"])
        self.associated = [a for a in c["associated"] if rng.random() < 0.45]
        self.red_present = [(r, rank) for r, rank in c["red_flags"] if rng.random() < (0.08 if rank == "emergency" else 0.15)]
        self.red_denied = []
        for d in c["denials"]:
            symptom = d.replace("no ", "").strip()
            if symptom in self.associated:
                continue
            if any(symptom.split()[0] in f for f, _ in self.red_present):
                continue
            if rng.random() < 0.55:
                self.red_denied.append(d)
        self.has_vitals = rng.random() < 0.55
        self.vitals = self._sample_vitals() if self.has_vitals else None
        self.meds = [self._sample_med() for _ in range(rng.randint(0, 3))]
        self.allergies = [a for a in ALLERGIES[:-1] if rng.random() < 0.08] or (
            ["None known"] if rng.random() < 0.3 else [])
        self.history = [h for h in HISTORY[:-1] if rng.random() < 0.18]
        self.skipped = set(rng.sample(["severity", "onset", "associated", "vitals", "meds",
                                       "allergies", "history", "redflags"], rng.randint(0, 3)))
        if "associated" in self.skipped:
            self.associated = []
        if "vitals" in self.skipped:
            self.vitals = None
        if "meds" in self.skipped:
            self.meds = []
        if "allergies" in self.skipped:
            self.allergies = []
        if "history" in self.skipped:
            self.history = []
        if "redflags" in self.skipped:
            self.red_present = []
            self.red_denied = []
        self.mention_phi = rng.random() < 0.4
        self.use_abbrev = rng.random() < 0.35
        self.use_hindi = self.lang == "mixed" and rng.random() < 0.35
        self._compute_urgency()

    def _sample_med(self):
        name, dose, freq, purpose, hints = MEDS[self.rng.randrange(len(MEDS))]
        return {"name": name, "dose": dose, "frequency": freq, "purpose": purpose, "hints": hints}

    def _sample_vitals(self):
        v = {}
        if self.rng.random() < 0.8:
            v["temp"] = self.rng.choice(["98.6F", "100.2F", "101.5F", "99.1F", "102.8F", "98.9F", "103.5F"])
        if self.rng.random() < 0.7:
            v["bp"] = self.rng.choice(["120/80", "135/90", "145/95", "118/76", "150/98"])
        if self.rng.random() < 0.7:
            v["hr"] = self.rng.choice([72, 80, 92, 104, 118])
        if self.rng.random() < 0.6:
            v["spo2"] = self.rng.choice([98, 97, 95, 93, 91])
        return v

    def _compute_urgency(self):
        c = COMPLAINTS[self.complaint_key]
        rank = {"routine": 0, "urgent": 1, "emergency": 2}
        order = ["routine", "urgent", "emergency"]
        u = order.index(c["base_urgency"])
        if "severity" not in self.skipped and self.severity == "severe":
            u = max(u, 1)
        if self.vitals:
            if self.vitals.get("spo2") is not None and self.vitals["spo2"] < 92:
                u = max(u, 2 if self.vitals["spo2"] < 90 else 1)
            if self.vitals.get("hr") is not None and self.vitals["hr"] > 120:
                u = max(u, 1)
            t = self.vitals.get("temp")
            if t and t.startswith("10") and float(t[:-1]) >= 103:
                u = max(u, 1)
            bp = self.vitals.get("bp")
            if bp:
                sys = int(bp.split("/")[0])
                dia = int(bp.split("/")[1])
                if sys >= 180 or dia >= 110:
                    u = max(u, 2)
        for _, flag_rank in self.red_present:
            u = max(u, order.index(flag_rank))
        self.urgency = order[u]
        self.escalate = self.urgency != "routine"


def _abbreviate(text, case):
    if not case.use_abbrev:
        return text
    for long, short in ABBREV.items():
        if long in text and case.rng.random() < 0.5:
            text = text.replace(long, short, 1)
    return text


def apply_noise(text, case):
    rng, level = case.rng, case.noise
    if level == 0:
        return text
    words = text.split()
    out = []
    p = 0.10 if level == 1 else 0.22
    for i, w in enumerate(words):
        if rng.random() < p:
            out.append(rng.choice(FILLERS))
        out.append(w)
        if rng.random() < p * 0.4:
            out.append(w)
        if i > 0 and rng.random() < p * 0.25:
            out.append(rng.choice(DISFLUENCIES).split(",")[0])
    return " ".join(out)


def _t(utterance, case, speaker):
    text = _abbreviate(utterance, case)
    if speaker == "Patient":
        text = apply_noise(text, case)
        if case.use_hindi and case.rng.random() < 0.5:
            text = f"{text}, {case.rng.choice(HINDI_FLAVOR[:4])}"
    return text


def render_transcript(case):
    c = COMPLAINTS[case.complaint_key]
    rng = case.rng
    lines = []

    def doc(t):
        lines.append(("Doctor", _t(t, case, "Doctor")))

    def pt(t):
        lines.append(("Patient", _t(t, case, "Patient")))

    def with_duration(phrase, duration, about=False):
        if duration.startswith("since"):
            return f"{phrase} {duration}."
        return f"{phrase} for {('about ' if about else '')}{duration}."

    doc(rng.choice(OPENERS))
    opening = case.phrase
    if case.mention_phi:
        opening = f"My name is [NAME], I am [AGE] years old. {case.phrase}"
    pt(with_duration(opening, case.duration, about=True))
    if "onset" not in case.skipped:
        doc("Did that start suddenly or come on gradually?")
        pt(f"It was {case.onset}.")
    if "severity" not in case.skipped:
        doc("How severe is it, on a scale of 1 to 10, mild, moderate, or severe?")
        pt(f"It's {case.severity}.")
    if "associated" not in case.skipped and case.associated:
        doc("Any other symptoms with it?")
        pt("Yes, " + ", and ".join(case.associated) + ".")
    if "redflags" not in case.skipped:
        for flag_text, _ in case.red_present:
            doc("Any " + flag_text.split()[0] + "?")
            pt(f"Yes, {flag_text}.")
        for denial in case.red_denied:
            doc("Any " + denial.replace("no ", "") + "?")
            pt(denial + ".")
    if "vitals" not in case.skipped and case.vitals:
        doc("Have you checked your vitals at home?")
        bits = []
        v = case.vitals
        if "temp" in v:
            bits.append(f"temperature {v['temp']}")
        if "bp" in v:
            bits.append(f"blood pressure {v['bp']}")
        if "hr" in v:
            bits.append(f"heart rate {v['hr']}")
        if "spo2" in v:
            bits.append(f"oxygen {v['spo2']}%")
        pt("Yes, " + ", ".join(bits) + ".")
    elif "vitals" not in case.skipped:
        doc("Have you checked your vitals at home?")
        pt("No, I haven't.")
    if "meds" not in case.skipped and case.meds:
        doc("What medications are you currently taking?")
        names = ", ".join(m["name"] for m in case.meds)
        pt(f"I take {names}.")
        doc("And the doses and how often?")
        pt(". ".join(f"{m['name']} {m['dose']}, {m['frequency']}" for m in case.meds) + ".")
    elif "meds" not in case.skipped:
        doc("Are you taking any medications?")
        pt("No, nothing right now.")
    if "allergies" not in case.skipped:
        doc("Any drug allergies?")
        if case.allergies and case.allergies != ["None known"]:
            pt("Yes, allergic to " + ", ".join(case.allergies) + ".")
        else:
            pt("No known allergies.")
    if "history" not in case.skipped and case.history:
        doc("Any medical history I should know about?")
        pt("I have " + ", ".join(case.history) + ".")
    closing = rng.choice([
        "Thanks, I'll take a look.",
        "Okay, noted. Anything else?",
        "Alright, thank you.",
    ])
    doc(closing)
    return "\n".join(f"{s}: {t}" for s, t in lines)


def render_gold(case):
    c = COMPLAINTS[case.complaint_key]
    rng = case.rng
    def with_duration(phrase, duration):
        if duration.startswith("since"):
            return f"{phrase} {duration}."
        return f"{phrase} for {duration}."

    def hpi_phrase(phrase):
        p = phrase.strip().strip(".")
        if p.startswith("I have "):
            return "having " + p[7:]
        if p.startswith("I'm ") or p.startswith("I am "):
            return "being " + p[p.find(" ") + 1:]
        if p.startswith("I can't "):
            return "unable to " + p[8:]
        if p.startswith("I "):
            return p[2:]
        if p.startswith("my "):
            return p[3:]
        return p

    parts = [with_duration(f"Patient reports {hpi_phrase(case.phrase)}", case.duration)]
    if "onset" not in case.skipped:
        parts.append(f"Onset was {case.onset}.")
    if "severity" not in case.skipped:
        parts.append(f"Severity described as {case.severity}.")
    if "associated" not in case.skipped and case.associated:
        parts.append("Associated symptoms: " + ", ".join(case.associated) + ".")
    for flag_text, _ in case.red_present:
        parts.append(f"Reports {flag_text}.")
    for denial in case.red_denied:
        parts.append("Denies " + denial.replace("no ", "") + ".")
    if case.history:
        parts.append("Past history: " + ", ".join(case.history) + ".")
    hpi = " ".join(parts)

    meds = []
    for m in case.meds:
        meds.append({"name": m["name"], "dose": m["dose"], "frequency": m["frequency"]})

    missing = []
    if "severity" in case.skipped:
        missing.append("severity not stated")
    if "onset" in case.skipped:
        missing.append("onset not stated")
    if "associated" in case.skipped:
        missing.append("associated symptoms not discussed")
    if "vitals" in case.skipped:
        missing.append("no vital signs recorded")
    if "meds" in case.skipped:
        missing.append("medication list not discussed")
    if "allergies" in case.skipped:
        missing.append("allergies not discussed")
    if "history" in case.skipped or not case.history:
        missing.append("past history not discussed")
    if "redflags" in case.skipped or (not case.red_present and not case.red_denied):
        missing.append("red flag symptoms not fully screened")

    vitals = {"temp": None, "bp": None, "hr": None, "spo2": None}
    if case.vitals:
        vitals.update({k: (str(v) if isinstance(v, float) else v) for k, v in case.vitals.items()})

    allergies = case.allergies if case.allergies and case.allergies != ["None known"] else []

    red_flags = [f for f, _ in case.red_present]

    return {
        "chief_complaint": f"{case.phrase} ({case.duration})",
        "hpi": hpi.strip(),
        "vitals": vitals,
        "medications": meds,
        "allergies": allergies,
        "red_flags": red_flags,
        "missing_info": missing,
        "urgency": case.urgency,
        "escalate": case.escalate,
    }


SYSTEM_PROMPT = (
    "You are a medical intake assistant running on a patient's phone. "
    "Extract the structured clinical summary from the chat transcript below. "
    "Only include information that is actually present in the transcript. "
    "If something was not discussed, put it in missing_info. "
    'urgency must be one of: "routine", "urgent", "emergency". '
    "escalate is true unless urgency is routine. "
    "Output strict JSON only, with exactly these keys: "
    "chief_complaint, hpi, vitals, medications, allergies, red_flags, missing_info, urgency, escalate. "
    "Do not output any text outside the JSON."
)


def render_example(case):
    transcript = render_transcript(case)
    gold = render_gold(case)
    return {
        "_id": case.qid,
        "noise": case.noise,
        "urgency": case.urgency,
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": "Transcript:\n" + transcript},
            {"from": "gpt", "value": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def generate(n, seed, noise, lang, start_id=0):
    rng = random.Random(seed)
    out = []
    keys = list(COMPLAINTS)
    for i in range(n):
        key = keys[i % len(keys)] if i < len(keys) else rng.choice(keys)
        case = Case(rng, key, noise, lang)
        ex = render_example(case)
        ex["_id"] = f"gen-{start_id + i:06d}"
        out.append(ex)
    return out


def build_aux(out_dir):
    topic_map = {}
    for key, c in COMPLAINTS.items():
        topic_map[key] = c["topics"]
    med_topics = {}
    for name, _, _, _, hints in MEDS:
        med_topics[name] = [name, "Medication information", "Side effects"]
    return topic_map, med_topics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--noise", type=int, default=2, choices=[0, 1, 2],
                    help="0=clean, 1=mild noise, 2=heavy noise")
    ap.add_argument("--lang", default="en", choices=["en", "mixed"])
    ap.add_argument("--quick", action="store_true", help="smoke test: 500/100")
    ap.add_argument("-o", "--out", default="dataset")
    args = ap.parse_args()

    n_train, n_val = (500, 100) if args.quick else (args.n_train, args.n_val)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = generate(n_train, args.seed, args.noise, args.lang, 0)
    val = generate(n_val, args.seed + 1, args.noise, args.lang, n_train)

    with open(out / "extraction_train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out / "extraction_val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out / "schema.json", "w") as f:
        json.dump(SCHEMA, f, indent=2)
    topic_map, med_topics = build_aux(out)
    with open(out / "topic_map.json", "w") as f:
        json.dump({"complaints": topic_map, "medications": med_topics}, f, indent=2)

    with open(out / "extraction_train.jsonl") as f:
        sample = json.loads(f.readline())
    from collections import Counter
    print(f"wrote {n_train} train + {n_val} val examples -> {out}/")
    mix = Counter(ex["urgency"] for ex in train)
    print("urgency mix (train):", dict(mix))
    print("sample example:\n", json.dumps(sample["conversations"][1]["value"], ensure_ascii=False)[:300],
          "\n---\n", sample["conversations"][2]["value"][:400])


if __name__ == "__main__":
    main()
