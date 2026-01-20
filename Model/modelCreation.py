# Run with this command: python Model\modelCreation.py --data_path "Data\dataset_final_v4.jsonl" --out_dir "ClinicalBERT_Run_v1" --epochs 5 --batch 8 --lr 2e-5
import os
import json
import argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

EMOTIONS = [
    "anxiety",
    "confusion",
    "frustration",
    "anger",
    "disappointment",
    "satisfaction",
]

def read_jsonl(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            emo = obj.get("emotions", {})
            y = [float(emo.get(e, 0)) for e in EMOTIONS]
            labels.append(y)
    return texts, labels

def make_dataset(texts, labels):
    return Dataset.from_dict({"text": texts, "labels": labels})

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_metrics_builder(threshold=0.5):
    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        probs = sigmoid(logits)
        y_pred = (probs >= threshold).astype(int)

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
        micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        neutral_true_ratio = float((y_true.sum(axis=1) == 0).mean())
        neutral_pred_ratio = float((y_pred.sum(axis=1) == 0).mean())

        return {
            "micro_f1": micro_f1,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "macro_f1": macro_f1,
            "neutral_true_ratio": neutral_true_ratio,
            "neutral_pred_ratio": neutral_pred_ratio,
        }
    return compute_metrics

def compute_pos_weight(train_labels: np.ndarray):
    pos = train_labels.sum(axis=0)
    neg = train_labels.shape[0] - pos
    pos_weight = np.ones_like(pos, dtype=np.float32)
    for i in range(len(pos)):
        if pos[i] > 0:
            pos_weight[i] = float(neg[i] / pos[i])
        else:
            pos_weight[i] = 1.0
    return torch.tensor(pos_weight, dtype=torch.float32)

class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        logits = outputs.logits
        labels = labels.to(dtype=torch.float32)
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"Data\dataset_final_v4.jsonl")
    parser.add_argument("--out_dir", type=str, default=r"ClinicalBERT_Model")
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    args.data_path = os.path.abspath(args.data_path)
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Loading data from: {args.data_path}")
    texts, labels = read_jsonl(args.data_path)
    indices = list(range(len(texts)))
    train_idx, val_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed)

    with open(os.path.join(args.out_dir, "split_indices.json"), "w") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx}, f)

    train_ds = make_dataset([texts[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = make_dataset([texts[i] for i in val_idx], [labels[i] for i in val_idx])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize_func(b):
        return tokenizer(b["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize_func, batched=True)
    val_ds = val_ds.map(tokenize_func, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(EMOTIONS), problem_type="multi_label_classification"
    )

    pos_weight = compute_pos_weight(np.array([labels[i] for i in train_idx], dtype=np.float32))

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_builder(args.threshold),
        pos_weight=pos_weight,
    )

    print("Starting training on ClinicalBERT...")
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    with open(os.path.join(args.out_dir, "emotions.json"), "w") as f:
        json.dump({"emotions": EMOTIONS, "threshold": args.threshold, "neutral_threshold": 0.35}, f)

    print(f"\nModel and settings saved to: {args.out_dir}")

if __name__ == "__main__":
    main()