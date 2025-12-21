# ===============================
# Baseline Model Evaluation Script
# ===============================

import os
import glob
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModel

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "clinical_emotisupport_dataset_numbered.jsonl"
MODEL_DIR = "./baseline_emotisupport"
BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 192
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD DATA
# -----------------------
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

emotion_keys = sorted(records[0]["emotions"].keys())
texts = [r["text"] for r in records]
labels = np.array([[r["emotions"][k] for k in emotion_keys] for r in records], dtype=np.float32)

idxs = list(range(len(texts)))
random.shuffle(idxs)
split = int(TRAIN_RATIO * len(idxs))
val_idxs = idxs[split:]

val_texts = [texts[i] for i in val_idxs]
val_labels = labels[val_idxs]

print(f"Loaded {len(records)} records")
print("Emotion labels:", emotion_keys)
print(f"Validation samples: {len(val_texts)}")

# -----------------------
# MODEL DEFINITION
# -----------------------
class EmotionRegressor(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.sigmoid(self.head(pooled))

# -----------------------
# LOAD TOKENIZER
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# -----------------------
# CREATE MODEL
# -----------------------
model = EmotionRegressor(BASE_MODEL_NAME, len(emotion_keys)).to(device)

# -----------------------
# FIND CHECKPOINT
# -----------------------
ckpts = sorted(
    glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")),
    key=lambda x: int(x.split("-")[-1])
)

if not ckpts:
    raise FileNotFoundError("No checkpoint-* folders found")

ckpt = ckpts[-1]
print("Using checkpoint:", ckpt)

# -----------------------
# LOAD WEIGHTS
# -----------------------
safetensor = os.path.join(ckpt, "model.safetensors")
binfile = os.path.join(ckpt, "pytorch_model.bin")

if os.path.exists(safetensor):
    from safetensors.torch import load_file
    state = load_file(safetensor)
elif os.path.exists(binfile):
    state = torch.load(binfile, map_location="cpu")
else:
    raise FileNotFoundError("No model weights found")

model.load_state_dict(state, strict=False)
model.eval()

# -----------------------
# PREDICT
# -----------------------
preds = []
with torch.no_grad():
    for t in val_texts:
        enc = tokenizer(
            t,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        p = model(enc["input_ids"], enc["attention_mask"]).cpu().numpy()[0]
        preds.append(p)

preds = np.array(preds)

# -----------------------
# METRICS
# -----------------------
mae = np.mean(np.abs(preds - val_labels))
rmse = np.sqrt(np.mean((preds - val_labels) ** 2))

pearsons = []
for i in range(len(emotion_keys)):
    if np.std(preds[:, i]) > 1e-6:
        pearsons.append(pearsonr(preds[:, i], val_labels[:, i])[0])
    else:
        pearsons.append(0.0)

print("\n=== FINAL METRICS ===")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("Pearson (mean):", round(np.mean(pearsons), 4))

# -----------------------
# PER-EMOTION MAE
# -----------------------
mae_per_emotion = {
    emotion_keys[i]: mean_absolute_error(val_labels[:, i], preds[:, i])
    for i in range(len(emotion_keys))
}

print("\nPer-emotion MAE:")
for k, v in mae_per_emotion.items():
    print(f"{k}: {v:.4f}")

# -----------------------
# PLOTS
# -----------------------
plt.figure(figsize=(8, 4))
plt.bar(mae_per_emotion.keys(), mae_per_emotion.values())
plt.xticks(rotation=45, ha="right")
plt.title("Per-Emotion MAE (Validation)")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

for emotion in ["anxiety", "confusion", "frustration", "anger", "disappointment", "satisfaction"]:
    if emotion in emotion_keys:
        i = emotion_keys.index(emotion)
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(val_labels[:, i], preds[:, i], alpha=0.75)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Predicted vs True — {emotion}")
        plt.tight_layout()
        plt.show()

print("\nDone.")
