# dataGenerator.py
from ollama import chat
import sys
import json
import time
import random
import re
from collections import Counter

# ===============================
# CONFIG
# ===============================
MODEL_NAME = "deepseek-r1:8b"
OUTPUT_FILE = "SIXTY_AND_MOVING_ON.jsonl"

TOTAL_SAMPLES = 500
BATCH_SIZE = 2
SEED = 42

MAX_RETRIES_PER_SAMPLE = 12
SLEEP_BETWEEN_BATCHES_SEC = 0.25

# anti-stall
FAIL_STREAK_SLEEP_AT = 25
FAIL_STREAK_SLEEP_SEC = 3.0

random.seed(SEED)

# ===============================
# ATTRIBUTE SPACES
# ===============================
DOMAINS = ["clinical", "administrative"]

CLINICAL_TOPICS = [
    "persistent pain",
    "worsening symptoms",
    "new unexplained symptoms",
    "unclear diagnosis",
    "conflicting medical opinions",
    "treatment not improving symptoms",
    "medication side effects",
    "medication dosage confusion",
    "treatment plan changes",
    "missed medication instructions",
    "waiting for test results",
    "delayed lab results",
    "lost test results",
    "unclear imaging findings",
    "repeat testing required",
    "difficulty reaching doctor",
    "follow-up appointment delay",
    "lack of post-treatment guidance",
    "discharge instructions unclear",
    "chronic condition management",
    "flare-up after improvement",
    "symptoms returning after treatment",
]

ADMIN_TOPICS = [
    "appointment scheduling issue",
    "appointment cancellation without notice",
    "long wait times",
    "rescheduled appointment multiple times",
    "no available appointments",
    "prescription renewal delay",
    "prescription sent to wrong pharmacy",
    "medication not available",
    "insurance approval for medication",
    "insurance coverage problem",
    "insurance claim denied",
    "unexpected billing charge",
    "billing discrepancy",
    "unclear payment responsibility",
    "portal access issue",
    "missing documents in patient portal",
    "messages not answered",
    "conflicting information from staff",
    "referral approval delay",
    "referral sent incorrectly",
    "specialist not accepting referral",
]

EMOTIONS = ["anxiety", "confusion", "frustration", "anger", "disappointment"]
AGE_GROUPS = ["young adult", "adult", "elderly"]

CHANNELS = ["phone_call", "patient_portal", "email", "in_person", "pharmacy"]
ACTORS = ["patient", "parent", "caregiver"]
RESOLUTION = ["unresolved", "delayed", "resolved"]
URGENCY = ["low", "medium", "high"]
LENGTH_BUCKETS = ["short", "medium", "long"]

STYLES = [
    "portal_message",
    "short_note",
    "detailed_story",
    "dialogue_snippet",
    "bullet_like",
    "followup_with_reference",
    "third_person_caregiver",
]

OPENERS = {
    "portal_message": [
        "On the patient portal, the status shows",
        "The portal now says",
        "I checked the portal this morning and",
        "In my portal messages, I can see",
        "The portal record for my visit shows",
    ],
    "short_note": [
        "Quick question:",
        "Need clarification:",
        "Following up:",
        "One issue:",
        "Checking on something:",
    ],
    "detailed_story": [
        "Last Thursday, I",
        "Over the past two weeks, I",
        "Since my appointment earlier this month, I",
        "After the visit, I",
        "This started about a week ago when I",
    ],
    "dialogue_snippet": [
        "The nurse said",
        "At the front desk they told me",
        "The pharmacist mentioned",
        "The receptionist told me",
        "During the call, I was told",
    ],
    "bullet_like": [
        "Summary:",
        "Details:",
        "Here are the main points:",
        "To clarify,",
        "What happened:",
    ],
    "followup_with_reference": [
        "Regarding my previous message,",
        "Following up on the request from last week,",
        "About the referral noted in my chart,",
        "Re: the issue from my last call,",
        "In reference to the appointment change,",
    ],
    "third_person_caregiver": [
        "I'm writing on behalf of my father.",
        "I'm contacting you for my mother.",
        "I'm reaching out for my spouse.",
        "I'm messaging for a family member in my care.",
        "I'm following up for an elderly relative.",
    ],
}

# NOTE:
# הדאטה שלכם בפועל *כן* מכיל מילים כמו confused/worried וכו'.
# כדי להמשיך "באותו אופן" אנחנו לא מפילים עליהן.
# לכן אין פה FORBIDDEN_WORDS FILTER.

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
- concise bullet-like lines separated by \\n (if style asks)
"""

# ===============================
# JSON EXTRACTION (fixes non-JSON wrappers)
# ===============================
def extract_first_json_object(s: str):
    if not s:
        raise ValueError("Empty model output")

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in output")

    candidate = s[start:end+1]
    return json.loads(candidate)

# ===============================
# QUOTA-AWARE CHOICE (soft)
# ===============================
TARGETS = {
    "domain": {"clinical": 0.50, "administrative": 0.50},
    "style": {s: 1.0 / len(STYLES) for s in STYLES},
    "channel": {c: 1.0 / len(CHANNELS) for c in CHANNELS},
    "age_group": {a: 1.0 / len(AGE_GROUPS) for a in AGE_GROUPS},
    "length_bucket": {"short": 0.30, "medium": 0.50, "long": 0.20},
    "emotion_marginal": {
        "anxiety": 0.32,
        "confusion": 0.30,
        "frustration": 0.35,
        "anger": 0.25,
        "disappointment": 0.30,
    },
}

def _soft_choose(options, counts: Counter, total_so_far: int, target_probs: dict):
    if total_so_far <= 0:
        return random.choice(options)

    deficits = []
    for opt in options:
        current = counts[opt] / float(total_so_far)
        target = target_probs.get(opt, 0.0)
        deficits.append(max(target - current, 0.0))

    if sum(deficits) == 0:
        return random.choice(options)

    weights = [d + 1e-6 for d in deficits]
    return random.choices(options, weights=weights, k=1)[0]

def _topic_options_for_domain(domain: str):
    return CLINICAL_TOPICS if domain == "clinical" else ADMIN_TOPICS

def _choose_emotion_pool(emotion_counts: Counter, total_so_far: int, k: int, exclude: set = None):
    exclude = exclude or set()
    options = [e for e in EMOTIONS if e not in exclude]
    if total_so_far <= 0:
        return random.sample(options, k=min(k, len(options)))

    weights = []
    for e in options:
        current = emotion_counts[e] / float(total_so_far) if total_so_far > 0 else 0.0
        target = TARGETS["emotion_marginal"].get(e, 0.0)
        deficit = max(target - current, 0.0)
        weights.append(deficit + 1e-6)

    if sum(weights) == 0:
        return random.sample(options, k=min(k, len(options)))

    chosen = []
    pool_options = options[:]
    pool_weights = weights[:]
    for _ in range(min(k, len(pool_options))):
        pick = random.choices(pool_options, weights=pool_weights, k=1)[0]
        idx = pool_options.index(pick)
        chosen.append(pick)
        pool_options.pop(idx)
        pool_weights.pop(idx)
    return chosen

# ===============================
# ATTRIBUTES
# ===============================
def sample_attributes(counters, total_so_far):
    domain = _soft_choose(DOMAINS, counters["domain"], total_so_far, TARGETS["domain"])
    topic = random.choice(_topic_options_for_domain(domain))

    style = _soft_choose(STYLES, counters["style"], total_so_far, TARGETS["style"])
    channel = _soft_choose(CHANNELS, counters["channel"], total_so_far, TARGETS["channel"])
    age_group = _soft_choose(AGE_GROUPS, counters["age_group"], total_so_far, TARGETS["age_group"])
    length_bucket = _soft_choose(LENGTH_BUCKETS, counters["length_bucket"], total_so_far, TARGETS["length_bucket"])

    actor = random.choice(ACTORS)
    resolution = random.choices(RESOLUTION, weights=[0.60, 0.30, 0.10], k=1)[0]
    urgency = random.choices(URGENCY, weights=[0.35, 0.45, 0.20], k=1)[0]

    pool_size = random.randint(2, 4)
    emotions_pool = _choose_emotion_pool(counters["emotion"], total_so_far, pool_size)

    emotions_pool = list(dict.fromkeys(emotions_pool))[:4]

    return {
        "domain": domain,
        "topic": topic,
        "style": style,
        "channel": channel,
        "actor": actor,
        "urgency": urgency,
        "resolution": resolution,
        "age_group": age_group,
        "length_bucket": length_bucket,
        "emotions_pool": emotions_pool,
    }

# ===============================
# PROMPT
# ===============================
def build_user_prompt(attrs):
    # pick 1-3 dominant emotions from pool
    num_emotions = random.randint(1, 3)
    dominant = random.sample(attrs["emotions_pool"], k=min(num_emotions, len(attrs["emotions_pool"])))

    emotions_map = {e: (1 if e in dominant else 0) for e in EMOTIONS}

    style = attrs["style"]
    opener_choices = OPENERS.get(style, ["Last week, I"])
    forced_openers = random.sample(opener_choices, k=min(3, len(opener_choices)))

    style_instructions = ""
    if style == "bullet_like":
        style_instructions = (
            "- Use 3-5 short lines separated by \\n.\n"
            "- Do NOT use markdown bullets like '-' or '*'. Plain lines only.\n"
        )
    elif style == "dialogue_snippet":
        style_instructions = (
            "- Include one quoted line of what staff said (e.g., \"...\").\n"
            "- Keep it short and realistic.\n"
        )
    elif style == "short_note":
        style_instructions = "- Keep it brief and direct.\n"
    elif style == "detailed_story":
        style_instructions = "- Include a mini timeline with at least 2 time references.\n"
    elif style == "followup_with_reference":
        style_instructions = (
            "- Refer to a prior contact.\n"
            "- Include a simple reference like a date or a small case number.\n"
        )
    elif style == "third_person_caregiver":
        style_instructions = "- Write as a caregiver about someone else (age/limitations ok).\n"
    elif style == "portal_message":
        style_instructions = "- Mention portal status / missing item / on-screen update.\n"

    return f"""
Generate ONE message that matches the dataset style seen in examples.

Attributes:
Domain: {attrs["domain"]}
Topic: {attrs["topic"]}
Channel: {attrs["channel"]}
Writer: {attrs["actor"]}
Urgency: {attrs["urgency"]}
Resolution: {attrs["resolution"]}
Patient age group: {attrs["age_group"]}
Length bucket: {attrs["length_bucket"]}
Style: {attrs["style"]}

START the message with ONE of these exact openers (choose one):
1) {forced_openers[0]}
2) {forced_openers[1] if len(forced_openers) > 1 else forced_openers[0]}
3) {forced_openers[2] if len(forced_openers) > 2 else forced_openers[0]}

Emotions (labels only, can be implied or even explicitly hinted in wording if needed):
{", ".join(dominant)}

Rules:
- Output EXACTLY ONE JSON object (no extra text).
- The JSON MUST contain keys: text, domain, language, emotions
- language MUST be "en"
- emotions MUST have exactly these keys: {", ".join(EMOTIONS)}
- emotions MUST match exactly this mapping (do not change it):
{json.dumps(emotions_map, ensure_ascii=False)}

Style constraints:
{style_instructions}

JSON schema:
{{
  "text": "...",
  "domain": "{attrs["domain"]}",
  "language": "en",
  "emotions": {json.dumps(emotions_map, ensure_ascii=False)}
}}
"""

# ===============================
# GENERATE
# ===============================
def generate_one(attrs):
    resp = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(attrs)},
        ],
        think=False,
        stream=False,
    )
    return resp.message.content.strip()

# ===============================
# COUNTERS
# ===============================
def update_counters(counters, attrs, data):
    counters["domain"][attrs["domain"]] += 1
    counters["style"][attrs["style"]] += 1
    counters["channel"][attrs["channel"]] += 1
    counters["age_group"][attrs["age_group"]] += 1
    counters["length_bucket"][attrs["length_bucket"]] += 1

    emos = data.get("emotions", {})
    for e in EMOTIONS:
        if int(emos.get(e, 0)) == 1:
            counters["emotion"][e] += 1

# ===============================
# LOAD STATE (continue IDs + avoid duplicates)
# ===============================
def load_existing_state(path):
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

# ===============================
# MAIN
# ===============================
def main():
    last_id, existing_texts, existing_count = load_existing_state(OUTPUT_FILE)

    # We want TOTAL_SAMPLES lines total in file.
    # If file already has some, we only generate the remaining.
    remaining = max(TOTAL_SAMPLES - existing_count, 0)

    if remaining == 0:
        print(f"File already has {existing_count} samples. Nothing to generate.")
        return

    current_id = last_id + 1
    generated_now = 0
    fail_streak = 0

    counters = {
        "domain": Counter(),
        "style": Counter(),
        "channel": Counter(),
        "age_group": Counter(),
        "length_bucket": Counter(),
        "emotion": Counter(),
    }

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while generated_now < remaining:
            for _ in range(BATCH_SIZE):
                if generated_now >= remaining:
                    break

                attrs = sample_attributes(counters, total_so_far=max(1, existing_count + generated_now))

                success = False
                last_err = None
                last_raw = None

                for _try in range(MAX_RETRIES_PER_SAMPLE):
                    try:
                        raw = generate_one(attrs)
                        last_raw = raw

                        data = extract_first_json_object(raw)

                        # enforce required fields
                        if "text" not in data or not isinstance(data["text"], str):
                            continue

                        text = data["text"].strip()
                        if not text:
                            continue

                        # avoid exact duplicates
                        if text in existing_texts:
                            continue

                        # enforce domain + language
                        data["domain"] = attrs["domain"]
                        data["language"] = "en"

                        # enforce emotions as exact 0/1 ints with correct keys
                        fixed = {}
                        for e in EMOTIONS:
                            v = data.get("emotions", {}).get(e, 0)
                            try:
                                v = int(v)
                            except Exception:
                                v = 0
                            fixed[e] = 1 if v == 1 else 0
                        data["emotions"] = fixed

                        # accept
                        data["id"] = current_id
                        current_id += 1

                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        existing_texts.add(text)

                        update_counters(counters, attrs, data)

                        generated_now += 1
                        success = True
                        fail_streak = 0
                        break

                    except Exception as e:
                        last_err = e
                        continue

                if not success:
                    fail_streak += 1
                    print("\n--- FAILED SAMPLE ---")
                    print("fail_streak:", fail_streak)
                    print("last_err:", repr(last_err))
                    if last_raw:
                        print("raw_preview:", last_raw[:500])
                    print("---------------------\n")

                    if fail_streak >= FAIL_STREAK_SLEEP_AT:
                        print(f"Too many failures in a row. Sleeping {FAIL_STREAK_SLEEP_SEC}s...")
                        time.sleep(FAIL_STREAK_SLEEP_SEC)
                        fail_streak = 0

            print(f"Generated {existing_count + generated_now}/{TOTAL_SAMPLES}")
            time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    print("Done. Dataset saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        print("Hint: run `ollama list` and verify the model name.")
        sys.exit(1)
