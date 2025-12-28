import argparse
import json
import os
import re
from collections import Counter, defaultdict

import matplotlib
import matplotlib.pyplot as plt

DEFAULT_DATA_PATH = "NLP\\Data\\dataset.jsonl"

# Keep in sync with NLP/Model/modelCreation.py EMOTIONS
EMOTION_KEYS = ["anxiety", "confusion", "frustration", "anger", "disappointment", "satisfaction"]
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


# Minimal, high-precision patterns to flag explicit emotion words.
# These are not used to label—only to detect mismatches / noisy supervision.
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


def save_bar_plot(path: str, title: str, xlabels: list, values: list, xlabel: str, ylabel: str):
    plt.figure(figsize=(9, 5))
    plt.bar(xlabels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EDA for dataset.jsonl (domains + multi-label emotions)")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", type=str, default=os.path.join("NLP", "Data", "eda_outputs"))
    parser.add_argument("--save_plots", action="store_true", help="Save plots to save_dir (PNG)")
    parser.add_argument("--no_show", action="store_true", help="Do not call plt.show()")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    data_path = args.data_path
    save_dir = args.save_dir

    if args.save_plots or args.no_show:
        matplotlib.use("Agg")

    os.makedirs(save_dir, exist_ok=True)

    # ================
    # Load data
    # ================
    records = []
    bad_lines = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception:
                bad_lines += 1

    total = len(records)
    print(f"Total messages: {total}")
    if bad_lines:
        print(f"Bad/unparseable lines skipped: {bad_lines}")

    if total == 0:
        raise SystemExit("No records loaded. Check --data_path and file format.")

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
    text_counter = Counter()

    # co-occurrence matrix counts
    co_counts = defaultdict(int)

    # Quality checks
    explicit_word_hits = Counter()  # how often each explicit-word pattern appears
    explicit_word_label_mismatch = Counter()  # word appears but label=0
    mismatch_examples = []

    for r in records:
        text = r.get("text", "") or ""
        domain = r.get("domain", "unknown")
        emotions = r.get("emotions", {}) or {}

        text_norm = text.strip()
        if text_norm:
            text_counter[text_norm] += 1

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
            bigram_counter[(filtered[i], filtered[i + 1])] += 1

        # co-occurrence
        for i in range(len(acts)):
            for j in range(i, len(acts)):
                a, b = acts[i], acts[j]
                key = (a, b) if a <= b else (b, a)
                co_counts[key] += 1

        # Explicit emotion-word mismatch (word present but label says 0)
        hits = detect_emotion_words(text)
        for h in hits:
            explicit_word_hits[h] += 1
            if int(emotions.get(h, 0) or 0) == 0:
                explicit_word_label_mismatch[h] += 1
                if len(mismatch_examples) < 12:
                    mismatch_examples.append(
                        {
                            "emotion": h,
                            "domain": domain,
                            "id": r.get("id"),
                            "text": text[:300],
                            "label_value": int(emotions.get(h, 0) or 0),
                        }
                    )

    # ================
    # Print headline stats
    # ================
    print("\n--- Domain distribution ---")
    for d in DOMAIN_KEYS:
        print(f"{d}: {domain_counter[d]} ({domain_counter[d] / total * 100:.1f}%)")
    unknown_domains = total - sum(domain_counter[d] for d in DOMAIN_KEYS)
    if unknown_domains:
        print(f"unknown: {unknown_domains}")

    print("\n--- Emotion frequency (how many messages contain the emotion) ---")
    for e in EMOTION_KEYS:
        c = emotion_counter[e]
        print(f"{e}: {c} ({c / total * 100:.1f}%)")

    print(f"\nMulti-label messages: {multi_label_count} ({multi_label_count / total * 100:.1f}%)")

    mean_len = sum(lengths_words) / len(lengths_words)
    print(f"\nMessage length (words) min/mean/max: {min(lengths_words)}/{mean_len:.1f}/{max(lengths_words)}")

    # Basic flags: too short / too long (tweak thresholds)
    SHORT_TH = 12
    LONG_TH = 140
    short_count = sum(1 for x in lengths_words if x < SHORT_TH)
    long_count = sum(1 for x in lengths_words if x > LONG_TH)
    print(f"Too short (<{SHORT_TH} words): {short_count} ({short_count / total * 100:.1f}%)")
    print(f"Too long (>{LONG_TH} words): {long_count} ({long_count / total * 100:.1f}%)")

    # Exact duplicates
    dup_texts = sum(1 for _t, c in text_counter.items() if c > 1)
    dup_rows = sum((c - 1) for _t, c in text_counter.items() if c > 1)
    print("\n--- Duplication (exact text match) ---")
    print(f"Unique texts: {len(text_counter)}")
    print(f"Duplicate texts (count>1): {dup_texts}")
    print(f"Duplicate rows beyond first: {dup_rows}")

    # Explicit emotion-word mismatches
    print("\n--- Explicit emotion-word vs label mismatches (word appears but label=0) ---")
    any_mismatch = False
    for e in EMOTION_KEYS:
        m = int(explicit_word_label_mismatch.get(e, 0))
        if m:
            any_mismatch = True
        print(f"{e}: {m} mismatches (word_hits={int(explicit_word_hits.get(e, 0))})")
    if mismatch_examples:
        print("\nExamples (up to 12):")
        for ex in mismatch_examples:
            print(f"- id={ex['id']} domain={ex['domain']} emotion={ex['emotion']} text={ex['text']}")
    if not any_mismatch:
        print("No explicit-word mismatches found (with current lexicon).")

    # ================
    # Plots (optional)
    # ================
    if args.save_plots:
        save_bar_plot(
            os.path.join(save_dir, "emotion_frequency.png"),
            "Emotion Frequency (messages where label=1)",
            EMOTION_KEYS,
            [emotion_counter[e] for e in EMOTION_KEYS],
            "Emotion",
            "Number of Messages",
        )
        save_bar_plot(
            os.path.join(save_dir, "domain_distribution.png"),
            "Domain Distribution",
            DOMAIN_KEYS,
            [domain_counter[d] for d in DOMAIN_KEYS],
            "Domain",
            "Number of Messages",
        )

        # Label cardinality
        max_k = max(label_cardinality_counter.keys())
        xs = list(range(0, max_k + 1))
        ys = [label_cardinality_counter.get(k, 0) for k in xs]
        plt.figure(figsize=(8, 5))
        plt.bar(xs, ys)
        plt.title("Multi-label Cardinality (how many emotions per message)")
        plt.xlabel("#Active emotions in message")
        plt.ylabel("Number of Messages")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "label_cardinality.png"), dpi=160)
        plt.close()

        # Length histogram
        plt.figure(figsize=(9, 5))
        plt.hist(lengths_words, bins=20)
        plt.title("Message Length Distribution (words)")
        plt.xlabel("Words per message")
        plt.ylabel("Count")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "length_hist_words.png"), dpi=160)
        plt.close()

        # Co-occurrence heatmap
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
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cooccurrence.png"), dpi=160)
        plt.close()

    if not args.no_show:
        # Keep interactive plots for local exploration.
        plt.figure(figsize=(9, 5))
        plt.bar(EMOTION_KEYS, [emotion_counter[e] for e in EMOTION_KEYS])
        plt.title("Emotion Frequency (messages where label=1)")
        plt.xlabel("Emotion")
        plt.ylabel("Number of Messages")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

        plt.figure(figsize=(7, 5))
        plt.bar(DOMAIN_KEYS, [domain_counter[d] for d in DOMAIN_KEYS])
        plt.title("Domain Distribution")
        plt.xlabel("Domain")
        plt.ylabel("Number of Messages")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

    # ================
    # Style / repetition diagnostics (prints)
    # ================
    print("\n--- Most common template openings ---")
    for label, c in template_counter.most_common(12):
        print(f"{label}: {c}")

    print("\n--- Most common unigrams (filtered) ---")
    for w, c in unigram_counter.most_common(args.top_k):
        print(f"{w}: {c}")

    print("\n--- Most common bigrams (filtered) ---")
    for (w1, w2), c in bigram_counter.most_common(args.top_k):
        print(f"{w1} {w2}: {c}")

    print("\n--- Repeated first sentences (potential duplication/templates) ---")
    for s, c in first_sentence_counter.most_common(10):
        if c >= 3:
            print(f"[{c}x] {s[:120]}{'...' if len(s) > 120 else ''}")

    report = {
        "data_path": data_path,
        "total": total,
        "bad_lines": bad_lines,
        "domain_counts": dict(domain_counter),
        "emotion_counts": {e: int(emotion_counter[e]) for e in EMOTION_KEYS},
        "multi_label_count": int(multi_label_count),
        "label_cardinality": {str(k): int(v) for k, v in label_cardinality_counter.items()},
        "length_words": {
            "min": int(min(lengths_words)),
            "mean": float(mean_len),
            "max": int(max(lengths_words)),
        },
        "duplication": {
            "unique_texts": int(len(text_counter)),
            "duplicate_texts": int(dup_texts),
            "duplicate_rows_beyond_first": int(dup_rows),
        },
        "explicit_word_hits": {e: int(explicit_word_hits.get(e, 0)) for e in EMOTION_KEYS},
        "explicit_word_label_mismatch": {e: int(explicit_word_label_mismatch.get(e, 0)) for e in EMOTION_KEYS},
        "mismatch_examples": mismatch_examples,
        "top_template_openings": template_counter.most_common(20),
        "top_unigrams": unigram_counter.most_common(50),
        "top_bigrams": [([w1, w2], int(c)) for (w1, w2), c in bigram_counter.most_common(50)],
    }

    out_path = os.path.join(save_dir, "eda_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nWrote EDA report:", out_path)


if __name__ == "__main__":
    main()
