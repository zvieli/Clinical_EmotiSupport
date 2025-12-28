import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    model_dir = r"Baseline Model\Model\distilbert_run"

    # Load emotions metadata
    meta_path = os.path.join(model_dir, "emotions.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    emotions = meta["emotions"]
    threshold = float(meta.get("threshold", 0.5))
    neutral_threshold = float(meta.get("neutral_threshold", 0.35))
    max_labels = int(meta.get("max_labels", 3))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    texts = [
        "I'm really worried because nobody answers the phone and my appointment is urgent.",
        "I don’t understand the results and the timeline keeps changing.",
        "Everything is fine, thank you for the great service!"
    ]

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()

    probs = sigmoid(logits)

    for i, text in enumerate(texts):
        scores = probs[i]

        # Sort emotions by confidence
        sorted_indices = np.argsort(scores)[::-1]
        top_scores = scores[sorted_indices]

        # If nothing is strong enough → Neutral
        if top_scores[0] < neutral_threshold:
            detected = ["neutral"]
        else:
            selected = []
            for idx in sorted_indices:
                if scores[idx] >= threshold and len(selected) < max_labels:
                    selected.append(emotions[idx])

            if not selected:
                detected = ["neutral"]
            else:
                detected = selected

        print("\nTEXT:", text)
        print("Detected emotions:", detected)
        print("Probabilities:", {emotions[j]: float(scores[j]) for j in range(len(emotions))})

if __name__ == "__main__":
    main()
