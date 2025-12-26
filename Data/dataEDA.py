import json
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

DATA_PATH = "Baseline Model\Data\dataset.jsonl"

EMOTION_KEYS = ["anxiety", "confusion", "frustration", "anger", "disappointment"]
DOMAIN_KEYS = ["clinical", "administrative"]

# ================
# Helpers
# ================
STOPWORDS = set([
    "i","me","my","mine","we","our","you","your","yours","he","she","they","them","their",
    "a","an","the","and","or","but","so","to","of","in","on","at","for","with","from","as",
    "is","are","was","were","be","been","being","do","does","did","have","has","had",
    "this","that","these","those","it","its","im","i'm","ive","i've","dont","don't","cant","can't",
    "not","no","yes","if","then","than","just","really","very","again","still"
])

def tokenize(text: str):
    # keep simple alphanum tokens
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return tokens

def first_sentence(text: str):
    # crude split for sentence end
    parts = re.split(r"[.!?]\s+", text.strip())
    return parts[0].strip() if parts else text.strip()

def template_prefix(text: str):
    """
    Detect common 'template-y' openings.
    This isn't perfect; it's used to spot repeated patterns.
    """
    t = text.strip().lower()

    patterns = [
        (r"^(i'?m|i am)\s+calling", "I am calling ..."),
        (r"^(i'?m|i am)\s+writing", "I am writing ..."),
        (r"^hello[, ]", "Hello ..."),
        (r"^hi[, ]", "Hi ..."),
        (r"^i\s+called", "I called ..."),
        (r"^i\s+have\s+been", "I have been ..."),
        (r"^i\s+need\s+to", "I need to ..."),
        (r"^i\s+was\s+scheduled", "I was scheduled ..."),
        (r"^i\s+received", "I received ..."),
        (r"^my\s+appointment", "My appointment ..."),
        (r"^the\s+pharmacy", "The pharmacy ..."),
        (r"^the\s+portal", "The portal ..."),
    ]

    for pat, label in patterns:
        if re.match(pat, t):
            return label

    # fallback: first 4 words
    words = t.split()
    return " ".join(words[:4]) + (" ..." if len(words) > 4 else "")

def active_emotions(emotions_dict: dict):
    # Your dataset uses 0/1 now
    return [e for e in EMOTION_KEYS if int(emotions_dict.get(e, 0)) == 1]

# ================
# Load data
# ================
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            records.append(rec)
        except Exception:
            continue

total = len(records)
print(f"Total messages: {total}")

if total == 0:
    raise SystemExit("No records loaded. Check DATA_PATH and file format.")

# ================
# Counters / Stats
# ================
emotion_counter = Counter()
domain_counter = Counter()
multi_label_count = 0
label_cardinality_counter = Counter()
lengths_words = []
lengths_chars = []
template_counter = Counter()
first_sentence_counter = Counter()
unigram_counter = Counter()
bigram_counter = Counter()

# co-occurrence matrix counts
co_counts = defaultdict(int)

for r in records:
    text = r.get("text", "")
    domain = r.get("domain", "unknown")
    emotions = r.get("emotions", {})

    domain_counter[domain] += 1

    acts = active_emotions(emotions)
    for e in acts:
        emotion_counter[e] += 1

    if len(acts) > 1:
        multi_label_count += 1

    label_cardinality_counter[len(acts)] += 1

    # lengths
    words = tokenize(text)
    lengths_words.append(len(words))
    lengths_chars.append(len(text))

    # style
    template_counter[template_prefix(text)] += 1
    first_sentence_counter[first_sentence(text)] += 1

    # ngrams (lightly filtered)
    filtered = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    unigram_counter.update(filtered)

    for i in range(len(filtered) - 1):
        bigram_counter[(filtered[i], filtered[i+1])] += 1

    # co-occurrence
    # count pairs among active emotions
    for i in range(len(acts)):
        for j in range(i, len(acts)):
            a, b = acts[i], acts[j]
            key = (a, b) if a <= b else (b, a)
            co_counts[key] += 1

# ================
# Print headline stats
# ================
print("\n--- Domain distribution ---")
for d in DOMAIN_KEYS:
    print(f"{d}: {domain_counter[d]} ({domain_counter[d]/total*100:.1f}%)")
unknown_domains = total - sum(domain_counter[d] for d in DOMAIN_KEYS)
if unknown_domains:
    print(f"unknown: {unknown_domains}")

print("\n--- Emotion frequency (how many messages contain the emotion) ---")
for e in EMOTION_KEYS:
    c = emotion_counter[e]
    print(f"{e}: {c} ({c/total*100:.1f}%)")

print(f"\nMulti-label messages: {multi_label_count} ({multi_label_count/total*100:.1f}%)")

mean_len = sum(lengths_words) / len(lengths_words)
print(f"\nMessage length (words) min/mean/max: {min(lengths_words)}/{mean_len:.1f}/{max(lengths_words)}")

# Basic flags: too short / too long (tweak thresholds)
SHORT_TH = 12
LONG_TH = 140
short_count = sum(1 for x in lengths_words if x < SHORT_TH)
long_count = sum(1 for x in lengths_words if x > LONG_TH)
print(f"Too short (<{SHORT_TH} words): {short_count} ({short_count/total*100:.1f}%)")
print(f"Too long (>{LONG_TH} words): {long_count} ({long_count/total*100:.1f}%)")

# ================
# Plots
# ================
# 1) Emotion frequency bar
plt.figure(figsize=(9, 5))
plt.bar(EMOTION_KEYS, [emotion_counter[e] for e in EMOTION_KEYS])
plt.title("Emotion Frequency (messages where label=1)")
plt.xlabel("Emotion")
plt.ylabel("Number of Messages")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# 2) Domain distribution
plt.figure(figsize=(7, 5))
plt.bar(DOMAIN_KEYS, [domain_counter[d] for d in DOMAIN_KEYS])
plt.title("Domain Distribution")
plt.xlabel("Domain")
plt.ylabel("Number of Messages")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# 3) Label cardinality distribution
max_k = max(label_cardinality_counter.keys())
xs = list(range(0, max_k + 1))
ys = [label_cardinality_counter.get(k, 0) for k in xs]

plt.figure(figsize=(8, 5))
plt.bar(xs, ys)
plt.title("Multi-label Cardinality (how many emotions per message)")
plt.xlabel("#Active emotions in message")
plt.ylabel("Number of Messages")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# 4) Length histogram (words)
plt.figure(figsize=(9, 5))
plt.hist(lengths_words, bins=20)
plt.title("Message Length Distribution (words)")
plt.xlabel("Words per message")
plt.ylabel("Count")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# 5) Emotion co-occurrence heatmap (matplotlib imshow)
# Build square matrix
matrix = [[0 for _ in EMOTION_KEYS] for __ in EMOTION_KEYS]
for i, a in enumerate(EMOTION_KEYS):
    for j, b in enumerate(EMOTION_KEYS):
        key = (a, b) if a <= b else (b, a)
        matrix[i][j] = co_counts.get(key, 0)

plt.figure(figsize=(8, 7))
plt.imshow(matrix, aspect="auto")
plt.title("Emotion Co-occurrence (counts)")
plt.xticks(range(len(EMOTION_KEYS)), EMOTION_KEYS, rotation=45, ha="right")
plt.yticks(range(len(EMOTION_KEYS)), EMOTION_KEYS)
plt.colorbar()
plt.show()

# ================
# Style / repetition diagnostics (prints)
# ================
print("\n--- Most common template openings ---")
for label, c in template_counter.most_common(12):
    print(f"{label}: {c}")

print("\n--- Most common unigrams (filtered) ---")
for w, c in unigram_counter.most_common(20):
    print(f"{w}: {c}")

print("\n--- Most common bigrams (filtered) ---")
for (w1, w2), c in bigram_counter.most_common(20):
    print(f"{w1} {w2}: {c}")

# Optional: find repeated first sentences (high repetition)
print("\n--- Repeated first sentences (potential duplication/templates) ---")
for s, c in first_sentence_counter.most_common(10):
    if c >= 3:  # tweak threshold
        print(f"[{c}x] {s[:120]}{'...' if len(s) > 120 else ''}")
