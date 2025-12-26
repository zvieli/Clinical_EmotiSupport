import os
import json
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

EMOTIONS = ["anxiety", "confusion", "frustration", "anger", "disappointment", "satisfaction"]

def read_jsonl(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            emo = obj["emotions"]
            texts.append(obj["text"])
            labels.append([float(emo.get(e, 0)) for e in EMOTIONS])
    return texts, np.array(labels, dtype=np.float32)

def make_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "labels": labels.tolist()})

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    # ====== EDIT THESE ======
    data_path = r"Baseline Model\Data\dataset.jsonl"
    model_dir = r"Baseline Model\Model\distilbert_run"
    max_length = 256
    test_size = 0.15
    seed = 42
    # ========================

    texts, labels = read_jsonl(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, shuffle=True
    )

    val_ds = make_dataset(X_val, y_val)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    val_ds = val_ds.map(tokenize, batched=True)
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # run inference
    all_logits = []
    all_true = []

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in loader:
            labels_b = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()

            all_logits.append(logits)
            all_true.append(labels_b)

    logits = np.vstack(all_logits)
    y_true = np.vstack(all_true).astype(int)

    probs = sigmoid(logits)

    best = None
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (probs >= thr).astype(int)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        micro_p  = precision_score(y_true, y_pred, average="micro", zero_division=0)
        micro_r  = recall_score(y_true, y_pred, average="micro", zero_division=0)

        if best is None or micro_f1 > best["micro_f1"]:
            best = {"threshold": float(thr), "micro_f1": float(micro_f1), "micro_precision": float(micro_p), "micro_recall": float(micro_r)}

    print("\nBEST THRESHOLD:")
    print(best)

    # Save threshold to emotions.json (keep in-sync with predict)
    emotions_path = os.path.join(model_dir, "emotions.json")
    payload = {"emotions": EMOTIONS, "threshold": best["threshold"]}
    with open(emotions_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated {emotions_path} with threshold={best['threshold']}")

if __name__ == "__main__":
    main()
