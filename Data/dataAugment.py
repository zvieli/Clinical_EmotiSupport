import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

from ollama import chat

# Keep in sync with NLP/Model/modelCreation.py EMOTIONS
EMOTIONS = ["anxiety", "confusion", "frustration", "anger", "disappointment", "satisfaction"]

DEFAULT_MODEL_NAME = "deepseek-r1:8b"
DEFAULT_OUT_FILE = os.path.join("NLP", "Data", "dataset.jsonl")

MAX_RETRIES_PER_SAMPLE = 12
SLEEP_BETWEEN_BATCHES_SEC = 0.25

SYSTEM_PROMPT = """
You are generating realistic patient / caregiver messages for a dataset.

Output rules:
- Output exactly ONE JSON object.
- No markdown.
- No explanations.
- No extra text outside the JSON.

The message can be short or long and may include:
- portal updates, phone call follow-ups, scheduling, billing, refills, referrals
- timelines, dates, case numbers
- concise bullet-like lines separated by \\n
Important:
- You MUST follow the exact emotions mapping provided in the user prompt.
""".strip()


# Minimal lexicon to catch explicit emotion wording that can confuse training labels.
# We intentionally keep it small and high-precision.
EMOTION_WORD_PATTERNS = {
    "anxiety": re.compile(
        r"\b(anxious|anxiety|worried|worry|concerned|panic|panicked|nervous|scared|afraid|stressed(?!\s+test)|stressed\s+out)\b",
        re.IGNORECASE,
    ),
    "confusion": re.compile(r"\b(confused|confusing|confusion|unclear|not\s+sure|unsure)\b", re.IGNORECASE),
    "frustration": re.compile(r"\b(frustrated|frustrating|frustration|stressful|overwhelmed)\b", re.IGNORECASE),
    "anger": re.compile(r"\b(angry|anger|mad|furious)\b", re.IGNORECASE),
    "disappointment": re.compile(r"\b(disappointed|disappointing|disappointment)\b", re.IGNORECASE),
    "satisfaction": re.compile(r"\b(satisfied|satisfaction|relieved|appreciate|appreciated|grateful|thank\s+you)\b", re.IGNORECASE),
}


NEUTRAL_OPENERS = [
    "Hello,",
    "Hi,",
    "Following up:",
    "Quick update:",
    "Checking on something:",
    "To confirm:",
    "Request:",
    "FYI:",
]

SATISFACTION_STYLES = [
    "appreciative",
    "professional_closure",
    "brief_confirmation",
    "calm_update",
    "caregiver_note",
]


def violates_explicit_emotion_words(text: str, emotions_map: dict):
    """Return (violations, hits) where:
    - violations: list of emotions whose label=0 but explicit words appear
    - hits: list of emotions whose explicit words appear
    """
    hits = []
    violations = []
    for emo, pattern in EMOTION_WORD_PATTERNS.items():
        if pattern.search(text or ""):
            hits.append(emo)
            if int(emotions_map.get(emo, 0) or 0) == 0:
                violations.append(emo)
    return violations, hits


def extract_first_json_object(s: str):
    if not s:
        raise ValueError("Empty model output")

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in output")

    candidate = s[start : end + 1]
    return json.loads(candidate)


def load_existing_state(path: str):
    last_id = 0
    existing_texts = set()
    existing_count = 0

    try:
        with open(path, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_count += 1
                    if isinstance(obj.get("id"), int):
                        last_id = max(last_id, obj["id"])
                    txt = obj.get("text")
                    if isinstance(txt, str) and txt.strip():
                        existing_texts.add(txt.strip())
                except Exception:
                    # ignore bad lines
                    pass
    except FileNotFoundError:
        pass

    return last_id, existing_texts, existing_count


def _ensure_emotions_map(emotions_map: dict):
    fixed = {}
    for e in EMOTIONS:
        v = int(emotions_map.get(e, 0) or 0)
        fixed[e] = 1 if v == 1 else 0
    return fixed


def build_prompt(kind: str, emotions_map: dict, domain: str):
    emotions_map = _ensure_emotions_map(emotions_map)

    opener = random.choice(NEUTRAL_OPENERS)

    if kind == "neutral_only":
        extra = (
            "Write a neutral, informational message. It should NOT express emotions (no worry, anger, frustration, etc.).\n"
            "It can be a simple status update, scheduling confirmation, billing clarification, or lab result availability.\n"
            "Avoid emotionally-loaded adjectives/adverbs.\n"
            "Avoid repeating common templates like 'Your appointment is scheduled...' too often; vary phrasing and structure.\n"
            f"Start with: {opener}\n"
        )
    elif kind == "gray_neutral":
        extra = (
            "Write a message that could be interpreted as potentially stressful, but the tone remains neutral and factual.\n"
            "Example situations: minor delay, pending document, rescheduling, clarification needed.\n"
            "IMPORTANT: do NOT use emotion words (worried, anxious, frustrated, angry, disappointed, upset, stressed, relieved, thankful).\n"
            "Do NOT add exclamation marks. Keep it calm and matter-of-fact.\n"
            f"Start with: {opener}\n"
        )
    elif kind == "satisfaction_only":
        extra = (
            "Write a clearly positive / relieved / appreciative message because something was resolved successfully.\n"
            "Do NOT include complaints or frustration.\n"
            "Vary phrasing: not every message should say 'thank you' or 'relieved'.\n"
            "Allow professional/neutral satisfaction too (e.g., 'Issue closed, no further action needed.').\n"
        )
    elif kind == "satisfaction_varied":
        style = random.choice(SATISFACTION_STYLES)
        if style == "appreciative":
            style_extra = (
                "Tone: appreciative and positive, but avoid cliché repetition.\n"
                "Use one gratitude cue at most, and avoid the phrase 'I'm so relieved'.\n"
            )
        elif style == "professional_closure":
            style_extra = (
                "Tone: professional closure. Example: 'The matter has been resolved as requested.'\n"
                "No exclamation marks. No emotional adjectives. Keep it concise.\n"
            )
        elif style == "brief_confirmation":
            style_extra = (
                "Tone: very brief confirmation. 1-2 sentences maximum.\n"
                "Example: 'Confirmed—issue resolved.'\n"
            )
        elif style == "calm_update":
            style_extra = (
                "Tone: calm, matter-of-fact, but with positive outcome.\n"
                "Mention what changed (approved/posted/scheduled/corrected) and that no further action is needed.\n"
            )
        else:  # caregiver_note
            style_extra = (
                "Write as a caregiver updating that the situation is now handled.\n"
                "Keep it realistic and short.\n"
            )

        extra = (
            "Write a satisfaction-positive message with high diversity.\n"
            "Do NOT always use 'thank you' and do NOT always use 'relieved'.\n"
            f"{style_extra}"
        )
    elif kind == "mixed_resolved":
        extra = (
            "Write a message where an issue was resolved (positive outcome), but the writer also references a prior difficulty.\n"
            "The tone should be balanced: resolution + lingering stress from the process.\n"
            "Keep it realistic: mention what was resolved and what was difficult.\n"
            "IMPORTANT: Do NOT use explicit emotion words for labels that are 0 (e.g., do not write 'anxious' if anxiety=0).\n"
            "Prefer factual wording (delays, back-and-forth, missing documents) instead of naming emotions.\n"
        )
    elif kind == "anger_only":
        extra = (
            "Write a message that clearly conveys anger/complaint about a mistake, unfair charge, repeated failures, or disrespectful handling.\n"
            "Use realistic complaint language (e.g., 'this is unacceptable', 'I need this escalated', 'I expect this to be corrected immediately').\n"
            "IMPORTANT: Do NOT use explicit emotion words for labels that are 0.\n"
            "Avoid saying 'frustrated', 'worried', 'confused', 'disappointed', or 'relieved' unless those labels are 1.\n"
            "You MAY use anger words because anger=1 for this kind.\n"
            "Keep it plausible for a patient/caregiver context (billing, scheduling, referral, refill, portal errors).\n"
        )
    elif kind == "anger_mixed":
        extra = (
            "Write a message that includes anger plus one additional negative emotion label (e.g., frustration or disappointment).\n"
            "The writer should sound escalated and demanding, but still within realistic patient/caregiver communication.\n"
            "IMPORTANT: Do NOT use explicit emotion words for labels that are 0.\n"
            "If frustration=1 you MAY use 'frustrated'; if disappointment=1 you MAY use 'disappointed'.\n"
            "Avoid 'anxious'/'worried' unless anxiety=1, and avoid 'confused' unless confusion=1.\n"
        )
    elif kind == "hard_negative":
        extra = (
            "Write a message that is realistic in this domain but stays emotionally neutral.\n"
            "It may mention logistics (timelines, missing documents, follow-up) but without emotional language.\n"
            "This is a 'hard negative': it can look similar to other messages but should not express emotions.\n"
            "Avoid repeating common templates; vary phrasing.\n"
            f"Start with: {opener}\n"
        )
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return f"""
Generate ONE message for a multi-label emotion dataset.

Kind: {kind}
Domain: {domain}

{extra}

Rules:
- Output EXACTLY ONE JSON object (no extra text).
- The JSON MUST contain keys: text, domain, language, emotions
- language MUST be \"en\"
- domain MUST be \"{domain}\"
- emotions MUST have exactly these keys: {", ".join(EMOTIONS)}
- emotions MUST match exactly this mapping (do not change it):
{json.dumps(emotions_map, ensure_ascii=False)}

JSON schema:
{{
  \"text\": \"...\",
  \"domain\": \"{domain}\",
  \"language\": \"en\",
  \"emotions\": {json.dumps(emotions_map, ensure_ascii=False)}
}}
""".strip()


def generate_one(model_name: str, prompt: str):
    resp = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        think=False,
        stream=False,
    )
    return resp.message.content.strip()


def sanitize_record(obj: dict, domain: str, emotions_map: dict):
    if "text" not in obj or not isinstance(obj["text"], str) or not obj["text"].strip():
        raise ValueError("Missing/empty text")

    record = {
        "text": obj["text"].strip(),
        "domain": domain,
        "language": "en",
        "emotions": _ensure_emotions_map(emotions_map),
    }

    violations, _hits = violates_explicit_emotion_words(record["text"], record["emotions"])
    if violations:
        raise ValueError(f"Explicit emotion words contradict labels: {violations}")
    return record


def main():
    parser = argparse.ArgumentParser(description="Append targeted synthetic examples to dataset.jsonl")
    parser.add_argument("--out_file", default=DEFAULT_OUT_FILE)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--add_neutral", type=int, default=0, help="Add all-zero emotion examples")
    parser.add_argument("--add_satisfaction", type=int, default=0, help="Add satisfaction-only examples")
    parser.add_argument("--add_hard_negatives", type=int, default=0, help="Add neutral hard negatives")
    parser.add_argument("--add_gray_neutral", type=int, default=0, help="Add gray-zone neutral examples (still all-zero)")
    parser.add_argument("--add_satisfaction_varied", type=int, default=0, help="Add diversified satisfaction-only examples")
    parser.add_argument(
        "--add_mixed_resolved",
        type=int,
        default=0,
        help="Add mixed examples: satisfaction + prior difficulty (e.g., satisfaction + frustration/anxiety)",
    )
    parser.add_argument("--add_anger", type=int, default=0, help="Add anger-only examples")
    parser.add_argument("--add_anger_mixed", type=int, default=0, help="Add anger + (frustration|disappointment) examples")

    parser.add_argument(
        "--domains",
        type=str,
        default="clinical,administrative",
        help="Comma-separated list; used for sampling domain",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--fresh", action="store_true", help="Overwrite output file")
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.fresh:
        try:
            os.remove(args.out_file)
        except FileNotFoundError:
            pass

    total_to_add = (
        int(args.add_neutral)
        + int(args.add_satisfaction)
        + int(args.add_hard_negatives)
        + int(args.add_gray_neutral)
        + int(args.add_satisfaction_varied)
        + int(args.add_mixed_resolved)
        + int(args.add_anger)
        + int(args.add_anger_mixed)
    )
    if total_to_add <= 0:
        print("Nothing to add. Use --add_neutral/--add_satisfaction/--add_hard_negatives")
        return

    last_id, existing_texts, existing_count = load_existing_state(args.out_file)
    current_id = last_id + 1

    domains = [d.strip() for d in (args.domains or "").split(",") if d.strip()]
    if not domains:
        domains = ["clinical", "administrative"]

    plan = []
    plan += [("neutral_only", args.add_neutral)]
    plan += [("satisfaction_only", args.add_satisfaction)]
    plan += [("satisfaction_varied", args.add_satisfaction_varied)]
    plan += [("hard_negative", args.add_hard_negatives)]
    plan += [("gray_neutral", args.add_gray_neutral)]
    plan += [("mixed_resolved", args.add_mixed_resolved)]
    plan += [("anger_only", args.add_anger)]
    plan += [("anger_mixed", args.add_anger_mixed)]
    plan = [(k, int(n)) for k, n in plan if int(n) > 0]

    def emotions_for_kind(kind: str):
        if kind == "neutral_only" or kind == "hard_negative":
            return {e: 0 for e in EMOTIONS}
        if kind == "gray_neutral":
            return {e: 0 for e in EMOTIONS}
        if kind == "satisfaction_only" or kind == "satisfaction_varied":
            return {**{e: 0 for e in EMOTIONS}, "satisfaction": 1}
        if kind == "mixed_resolved":
            # Keep satisfaction=1 but add one mild negative label to create a gray boundary.
            extra = random.choice(["frustration", "anxiety", "disappointment"])  # keep realistic
            m = {e: 0 for e in EMOTIONS}
            m["satisfaction"] = 1
            m[extra] = 1
            return m
        if kind == "anger_only":
            return {**{e: 0 for e in EMOTIONS}, "anger": 1}
        if kind == "anger_mixed":
            extra = random.choice(["frustration", "disappointment"])
            m = {e: 0 for e in EMOTIONS}
            m["anger"] = 1
            m[extra] = 1
            return m
        raise ValueError(kind)

    counters = Counter()
    failed = 0

    with open(args.out_file, "a", encoding="utf-8") as wf:
        remaining_total = total_to_add

        for kind, count in plan:
            emotions_map = emotions_for_kind(kind)

            added_for_kind = 0
            while added_for_kind < count:
                batch = min(args.batch_size, count - added_for_kind)

                for _ in range(batch):
                    domain = random.choice(domains)
                    prompt = build_prompt(kind=kind, emotions_map=emotions_map, domain=domain)

                    success = False
                    last_raw = None
                    last_err = None

                    for _try in range(MAX_RETRIES_PER_SAMPLE):
                        try:
                            raw = generate_one(args.model_name, prompt)
                            last_raw = raw
                            obj = extract_first_json_object(raw)
                            rec = sanitize_record(obj, domain=domain, emotions_map=emotions_map)

                            if rec["text"] in existing_texts:
                                continue

                            rec["id"] = current_id
                            current_id += 1

                            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            existing_texts.add(rec["text"])

                            added_for_kind += 1
                            remaining_total -= 1
                            counters[kind] += 1
                            success = True
                            break
                        except Exception as e:
                            last_err = e
                            continue

                    if not success:
                        failed += 1
                        print("\n--- FAILED SAMPLE ---")
                        print("kind:", kind)
                        print("last_err:", repr(last_err))
                        if last_raw:
                            print("raw_preview:", last_raw[:500])
                        print("---------------------\n")

                print(
                    f"Progress: wrote {existing_count + (current_id - (last_id + 1))} total lines | "
                    f"remaining to add: {remaining_total}"
                )
                time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    report = {
        "out_file": args.out_file,
        "existing_count_before": existing_count,
        "added": dict(counters),
        "failed": failed,
        "seed": args.seed,
        "model_name": args.model_name,
    }
    report_path = os.path.splitext(args.out_file)[0] + ".augment_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("Wrote report to:", report_path)
    except Exception:
        pass

    print("Done. Appended to:", args.out_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        print("Hint: ensure Ollama is running and the model exists (try: ollama list).")
        sys.exit(1)
