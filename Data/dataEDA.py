import argparse
import json
import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Arguments
DEFAULT_DATA_PATH = r"Data\dataset_final_v4.jsonl"
EMOTION_KEYS = ["anxiety", "confusion", "frustration", "anger", "disappointment", "satisfaction"]
DOMAIN_KEYS = ["clinical", "administrative"]

def active_emotions(emotions_dict: dict):
    return [e for e in EMOTION_KEYS if int(emotions_dict.get(e, 0)) == 1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", type=str, default=os.path.join("Data", "eda_outputs"))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    records = []
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File not found at {args.data_path}")
        return

    total = len(records)
    print(f"Total messages loaded: {total}")

    emotion_counter = Counter()
    domain_counter = Counter()
    lengths_words = []
    label_cardinality = Counter()
    co_counts = defaultdict(int)

    for r in records:
        domain_counter[r.get("domain", "unknown")] += 1
        
        acts = active_emotions(r.get("emotions", {}))
        label_cardinality[len(acts)] += 1
        
        for e in acts:
            emotion_counter[e] += 1
        
        txt = r.get("text", "")
        lengths_words.append(len(txt.split()))

        for i in range(len(acts)):
            for j in range(i, len(acts)):
                a, b = acts[i], acts[j]
                key = (a, b) if a <= b else (b, a)
                co_counts[key] += 1

    print("Statistics processed. Generating plots...")

# Graphs

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(EMOTION_KEYS, [emotion_counter[e] for e in EMOTION_KEYS], color='skyblue')
    ax1.set_title("1. Emotion Frequency")
    ax1.set_ylabel("Count")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    fig1.savefig(os.path.join(args.save_dir, "1_emotions.png"))
    print("Saved 1_emotions.png")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    domains = list(domain_counter.keys())
    counts = [domain_counter[d] for d in domains]
    ax2.bar(domains, counts, color='salmon')
    ax2.set_title("2. Domain Distribution")
    fig2.savefig(os.path.join(args.save_dir, "2_domains.png"))
    print("Saved 2_domains.png")

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sorted_card = sorted(label_cardinality.items())
    x_labels = [str(item[0]) for item in sorted_card]
    y_values = [item[1] for item in sorted_card]
    
    ax3.bar(x_labels, y_values, color='plum')
    ax3.set_title("3. Emotions per Message (Cardinality)")
    ax3.set_xlabel("Number of Labels")
    ax3.set_ylabel("Message Count")
    fig3.savefig(os.path.join(args.save_dir, "3_cardinality.png"))
    print("Saved 3_cardinality.png")

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(lengths_words, bins=30, color='lightgreen', edgecolor='black')
    ax4.set_title("4. Message Length Distribution")
    ax4.set_xlabel("Words per Message")
    ax4.set_ylabel("Frequency")
    fig4.savefig(os.path.join(args.save_dir, "4_lengths.png"))
    print("Saved 4_lengths.png")

    fig5, ax5 = plt.subplots(figsize=(10, 8))
    matrix = np.zeros((len(EMOTION_KEYS), len(EMOTION_KEYS)))
    for i, e1 in enumerate(EMOTION_KEYS):
        for j, e2 in enumerate(EMOTION_KEYS):
            key = (e1, e2) if e1 <= e2 else (e2, e1)
            matrix[i, j] = co_counts.get(key, 0)

    im = ax5.imshow(matrix, cmap='YlOrRd')
    ax5.set_title("5. Emotion Co-occurrence Heatmap")
    
    ax5.set_xticks(np.arange(len(EMOTION_KEYS)))
    ax5.set_yticks(np.arange(len(EMOTION_KEYS)))
    ax5.set_xticklabels(EMOTION_KEYS, rotation=45, ha="right")
    ax5.set_yticklabels(EMOTION_KEYS)
    
    for i in range(len(EMOTION_KEYS)):
        for j in range(len(EMOTION_KEYS)):
            text_color = "white" if matrix[i, j] > matrix.max()/2 else "black"
            ax5.text(j, i, int(matrix[i, j]), ha="center", va="center", color=text_color)

    fig5.colorbar(im, ax=ax5)
    fig5.tight_layout()
    fig5.savefig(os.path.join(args.save_dir, "5_heatmap.png"))
    print("Saved 5_heatmap.png")

    print("\n--- DONE ---")
    print("Attempting to show plots. Check your taskbar if windows don't appear on top.")
    plt.show()

    all_openers = []
    for r in records:
        words = r.get("text", "").split()
        if len(words) >= 4:
            opener = " ".join(words[:4])
            all_openers.append(opener)
    
    top_openers = Counter(all_openers).most_common(10)
    opener_labels = [x[0] for x in top_openers]
    opener_counts = [x[1] for x in top_openers]

    fig6, ax6 = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(opener_labels))
    ax6.barh(y_pos, opener_counts, color='teal')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(opener_labels)
    ax6.invert_yaxis() 
    ax6.set_xlabel('Frequency')
    ax6.set_title('6. Top 10 Frequent Openers (Data Diversity Check)')


    for i, v in enumerate(opener_counts):
        ax6.text(v + 1, i, str(v), va='center')

    fig6.tight_layout()
    fig6.savefig(os.path.join(args.save_dir, "6_openers.png"))
    print("Saved 6_openers.png")

    print("\n--- DONE ---")
    print("Attempting to show plots...")
    plt.show()

if __name__ == "__main__":
    main()