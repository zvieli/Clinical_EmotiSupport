import json
from collections import Counter
import matplotlib.pyplot as plt

# Path to your dataset
DATA_PATH = "clinical_emotisupport_dataset_numbered.jsonl"

emotion_counter = Counter()

# Load data and count emotions
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        emotions = record["emotions"]
        for emotion, value in emotions.items():
            if value > 0:
                emotion_counter[emotion] += 1

# Sort emotions by frequency
emotions = list(emotion_counter.keys())
counts = list(emotion_counter.values())

# Plot
plt.figure(figsize=(8, 5))
plt.bar(emotions, counts)
plt.title("Distribution of Emotions (Frequency)")
plt.xlabel("Emotion")
plt.ylabel("Number of Messages")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

plt.show()
