from ollama import chat
import sys
import uuid
import json
import time

MODEL_NAME = "deepseek-r1:8b"
OUTPUT_FILE = "clinical_emotisupport_dataset.jsonl"
TOTAL_SAMPLES = 100
BATCH_SIZE = 5

SYSTEM_PROMPT = """
You are generating realistic patient messages for a machine learning dataset called “Clinical EmoTiSupport”.

Your task has TWO strictly separated phases:

PHASE 1 — Patient Message:
- Write a message containing ONLY facts, events, questions, actions, timelines, and decisions.
- The message must sound like something a real patient would send.

ABSOLUTE RULE (NO EXCEPTIONS):
- The patient text MUST NOT mention emotions in ANY way.
- Do NOT use emotional words, mental states, or feeling-related phrasing.

Forbidden examples (NOT allowed):
worried, anxious, stressed, overwhelmed, scared, afraid, confused, frustrated,
upset, disappointed, relieved, happy, satisfied, nervous, concerned,
“I feel”, “I’m feeling”, “it makes me”, “I feel like”, “I don’t know how to feel”.

If the text describes an internal emotional state — the output is INVALID.

How to express situations WITHOUT emotions:
- Uncertainty → missing information, unclear instructions, delayed results.
- Pressure → long symptom duration, deadlines, repeated attempts.
- Seriousness → abnormal results, escalation, worsening symptoms.
- Do NOT hint at feelings using emotional adjectives.

PHASE 2 — Emotion Annotation:
- Emotions appear ONLY in the emotion vector.
- At least ONE dominant emotion MUST be present (>= 0.60).
- Multi-label emotions allowed but realistic.
- Administrative messages usually include frustration or confusion.
- Clinical messages usually include anxiety or confusion.
- Satisfaction should be low unless clearly resolved.

General rules:
- Avoid repeating scenarios, doctors, conditions, or wording.
- Each message must be clearly distinct from previous ones.
- Use everyday, imperfect language — not formal or clinical writing.

Domains:
- "clinical": symptoms, tests, diagnoses, treatments, side effects, recovery.
- "administrative": appointments, billing, insurance, portals, access issues.

Output rules:
- One valid JSON object per line (JSONL).
- No explanations.
- No markdown.
- No extra text.

Schema:
{
  "text": "...",
  "domain": "clinical" | "administrative",
  "language": "en",
  "emotions": {
    "anxiety": 0.00,
    "confusion": 0.00,
    "frustration": 0.00,
    "anger": 0.00,
    "disappointment": 0.00,
    "satisfaction": 0.00
  }
}
"""

USER_PROMPT = """
Generate 5 patient messages following all instructions.

Requirements:
- Roughly half clinical and half administrative.
- Vary length, urgency, and structure.
- Include realistic ambiguity and incomplete information.
- Cover a mix of mild issues, ongoing problems, and at least one serious case.
- Do NOT reuse scenarios, wording, or medical issues within the batch.
"""

def generate_batch():
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        think=False,
        stream=False,
    )
    return response.message.content.strip().splitlines()

def main():
    generated = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while generated < TOTAL_SAMPLES:
            lines = generate_batch()

            for line in lines:
                if generated >= TOTAL_SAMPLES:
                    break

                try:
                    data = json.loads(line)
                    data["id"] = str(uuid.uuid4())
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    generated += 1
                except json.JSONDecodeError:
                    continue

            print(f"Generated {generated}/{TOTAL_SAMPLES}")
            time.sleep(0.5)

    print("Done. Dataset saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        print("Hint: run ollama list and verify the model name.")
        sys.exit(1)