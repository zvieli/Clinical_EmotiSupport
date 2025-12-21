import json
import random
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

from scipy.stats import pearsonr


# -----------------------
# Config
# -----------------------
DATA_PATH = "clinical_emotisupport_dataset_numbered.jsonl"
MODEL_NAME = "distilbert-base-uncased"  # baseline; can switch to bert-base-uncased / roberta-base / etc.
MAX_LEN = 192
SEED = 42

TRAIN_RATIO = 0.8
BATCH_SIZE = 8
EPOCHS = 8  # small dataset => a few epochs OK
LR = 2e-5
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "./baseline_emotisupport"

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------
# Load JSONL
# -----------------------
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

assert len(records) > 0, "Dataset is empty."

# Collect emotion keys (assumes consistent schema across records)
emotion_keys = sorted(records[0]["emotions"].keys())
num_labels = len(emotion_keys)

print("Loaded records:", len(records))
print("Emotion labels:", emotion_keys)

# Create arrays
texts = [r["text"] for r in records]
y = np.array([[float(r["emotions"][k]) for k in emotion_keys] for r in records], dtype=np.float32)

# Shuffle and split
idxs = list(range(len(records)))
random.shuffle(idxs)

train_size = int(TRAIN_RATIO * len(idxs))
train_idxs = idxs[:train_size]
val_idxs = idxs[train_size:]

train_texts = [texts[i] for i in train_idxs]
val_texts = [texts[i] for i in val_idxs]
train_y = y[train_idxs]
val_y = y[val_idxs]

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")


# -----------------------
# Dataset
# -----------------------
class EmotiSupportDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.float32)
        return item


# -----------------------
# Model: Transformer encoder + regression head
# -----------------------
class EmotionRegressor(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.reg_head = nn.Linear(hidden, num_labels)
        self.sigmoid = nn.Sigmoid()

        # MSE is a solid baseline for intensity regression
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT: outputs.last_hidden_state shape [B, T, H] ; take [CLS] token (position 0)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.reg_head(pooled)
        preds = self.sigmoid(logits)  # constrain to [0,1]

        loss = None
        if labels is not None:
            loss = self.loss_fn(preds, labels)

        return {"loss": loss, "logits": preds}


# -----------------------
# Metrics
# -----------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.clip(preds, 0.0, 1.0)
    labels = np.clip(labels, 0.0, 1.0)

    # Overall MAE/RMSE (averaged over all labels and samples)
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))

    # Pearson per emotion (skip if constant)
    pearsons = []
    per_label = {}
    for j, emo in enumerate(emotion_keys):
        x = preds[:, j]
        t = labels[:, j]
        if np.std(x) < 1e-8 or np.std(t) < 1e-8:
            r = 0.0
        else:
            r = pearsonr(x, t)[0]
            if np.isnan(r):
                r = 0.0
        pearsons.append(r)
        per_label[f"pearson_{emo}"] = r

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "pearson_mean": float(np.mean(pearsons)),
    }
    # Uncomment if you want per-label pears in logs (can be noisy on slide)
    # metrics.update(per_label)
    return metrics


# -----------------------
# Train
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = EmotiSupportDataset(train_texts, train_y, tokenizer, MAX_LEN)
val_ds = EmotiSupportDataset(val_texts, val_y, tokenizer, MAX_LEN)

model = EmotionRegressor(MODEL_NAME, num_labels)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="mae",
    greater_is_better=False,
    report_to="none",
    fp16=torch.cuda.is_available(),
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nFinal eval:")
metrics = trainer.evaluate()
print(metrics)

# Save final model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nSaved to:", OUTPUT_DIR)
print("\nEmotion order:", emotion_keys)
