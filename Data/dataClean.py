import argparse
import json
import os
import re
from collections import Counter

DEFAULT_IN_FILE = os.path.join("NLP", "Data", "dataset.jsonl")
DEFAULT_OUT_FILE = os.path.join("NLP", "Data", "dataset.cleaned.jsonl")

# Keep in sync with NLP/Model/modelCreation.py EMOTIONS
EMOTION_KEYS = ["anxiety", "confusion", "frustration", "anger", "disappointment", "satisfaction"]
DOMAIN_KEYS = {"clinical", "administrative"}

# High-precision patterns: enforce consistency (word present => label=1).
EMOTION_WORD_PATTERNS = {
    "anxiety": re.compile(
        r"\b(anxious|anxiety|worried|worry|concerned|panic|panicked|nervous|scared|afraid|stressed(?!\s+test)|stressed\s+out)\b",
        re.IGNORECASE,
    ),
    "confusion": re.compile(r"\b(confused|confusing|confusion|unclear|not\s+sure|unsure)\b", re.IGNORECASE),
    "frustration": re.compile(r"\b(frustrated|frustrating|frustration|stressful|overwhelmed)\b", re.IGNORECASE),
    "anger": re.compile(r"\b(angry|anger|mad|furious)\b", re.IGNORECASE),
    "disappointment": re.compile(r"\b(disappointed|disappointing|disappointment)\b", re.IGNORECASE),
    "satisfaction": re.compile(
        r"\b(satisfied|satisfaction|relieved|appreciate|appreciated|grateful|thank\s+you)\b",
        re.IGNORECASE,
    ),
}


def detect_emotion_words(text: str):
    hits = []
    for emo, pat in EMOTION_WORD_PATTERNS.items():
        if pat.search(text or ""):
            hits.append(emo)
    return hits


def normalize_record(rec: dict):
    rec = dict(rec)
    rec["text"] = (rec.get("text") or "").strip()
    rec["domain"] = rec.get("domain") or "unknown"
    rec["language"] = rec.get("language") or "en"

    emotions = rec.get("emotions") or {}
    emotions = dict(emotions)

    for k in EMOTION_KEYS:
        v = emotions.get(k, 0)
        try:
            v = int(v)
        except Exception:
            v = 0
        emotions[k] = 1 if v == 1 else 0

    rec["emotions"] = emotions
    return rec


def main():
    p = argparse.ArgumentParser(
        description="Clean dataset.jsonl by enforcing label consistency with explicit emotion words (word => label=1)."
    )
    p.add_argument("--in_file", type=str, default=DEFAULT_IN_FILE)
    p.add_argument("--out_file", type=str, default=DEFAULT_OUT_FILE)
    p.add_argument(
        "--mode",
        type=str,
        choices=["relabel", "drop"],
        default="relabel",
        help="relabel: set label=1 when explicit word appears; drop: remove rows with mismatches",
    )
    p.add_argument("--write_report", action="store_true", help="Write a .clean_report.json next to out_file")
    p.add_argument(
        "--overwrite_in_place",
        action="store_true",
        help="Overwrite in_file (creates .bak copy first). Use with caution.",
    )
    args = p.parse_args()

    in_file = args.in_file
    out_file = args.out_file

    if args.overwrite_in_place:
        out_file = in_file

    raw = []
    bad = 0
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except Exception:
                bad += 1

    total = len(raw)
    if total == 0:
        raise SystemExit(f"No records loaded from: {in_file}")

    if args.overwrite_in_place:
        bak = in_file + ".bak"
        if not os.path.exists(bak):
            with open(bak, "w", encoding="utf-8") as fb:
                with open(in_file, "r", encoding="utf-8") as fa:
                    fb.write(fa.read())

    changed_rows = 0
    dropped_rows = 0
    relabel_counts = Counter()
    mismatch_counts = Counter()
    domain_unknown = 0

    cleaned = []

    for rec in raw:
        rec = normalize_record(rec)
        if rec["domain"] not in DOMAIN_KEYS:
            domain_unknown += 1

        text = rec["text"]
        emotions = rec["emotions"]

        hits = detect_emotion_words(text)
        mismatches = [h for h in hits if emotions.get(h, 0) == 0]

        for h in mismatches:
            mismatch_counts[h] += 1

        if mismatches and args.mode == "drop":
            dropped_rows += 1
            continue

        if mismatches and args.mode == "relabel":
            for h in mismatches:
                emotions[h] = 1
                relabel_counts[h] += 1
            rec["emotions"] = emotions
            changed_rows += 1

        cleaned.append(rec)

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in cleaned:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "in_file": in_file,
        "out_file": out_file,
        "mode": args.mode,
        "total_in": total,
        "bad_lines_skipped": bad,
        "total_out": len(cleaned),
        "rows_changed": changed_rows,
        "rows_dropped": dropped_rows,
        "domain_unknown_rows": domain_unknown,
        "mismatch_counts_before_fix": {k: int(mismatch_counts.get(k, 0)) for k in EMOTION_KEYS},
        "relabel_counts_applied": {k: int(relabel_counts.get(k, 0)) for k in EMOTION_KEYS},
    }

    if args.write_report:
        rep_path = out_file + ".clean_report.json"
        with open(rep_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, ensure_ascii=False, indent=2)
        print("Wrote report:", rep_path)

    print("Wrote cleaned dataset:", out_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
