import os
import json
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_predictions(probs_row, emotions, threshold, neutral_threshold, max_labels):
    """
    Implements:
    - neutral_threshold: if max prob < neutral_threshold => "none"
    - threshold: select labels with prob >= threshold
    - max_labels: cap number of predicted labels to top-k by probability
    - fallback: if not neutral but nothing passed threshold => take top-1
    """
    probs_row = np.array(probs_row, dtype=float)
    max_prob = float(probs_row.max())

    # Neutral gate
    if max_prob < neutral_threshold:
        return []

    # Normal thresholding
    idxs = np.where(probs_row >= threshold)[0].tolist()

    # Fallback: not neutral but nothing passed threshold => choose top-1
    if len(idxs) == 0:
        idxs = [int(np.argmax(probs_row))]

    # Cap to max_labels by probability
    if len(idxs) > max_labels:
        idxs = sorted(idxs, key=lambda i: probs_row[i], reverse=True)[:max_labels]

    return idxs


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a locally-saved multi-label Transformers model")
    parser.add_argument("--model_dir", type=str, default=r"Baseline Model\Model\distilbert_run")
    parser.add_argument("--data_path", type=str, default=r"Data\dataset.jsonl")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override label threshold (default: read from emotions.json)",
    )
    parser.add_argument(
        "--neutral_threshold",
        type=float,
        default=None,
        help="Override neutral gate (default: read from emotions.json)",
    )
    parser.add_argument(
        "--max_labels",
        type=int,
        default=None,
        help="Override max labels per example (default: read from emotions.json)",
    )
    parser.add_argument(
        "--scan_neutral_threshold",
        action="store_true",
        help="Scan neutral_threshold over a range and pick the best by neutral F1 (val split recommended)",
    )
    parser.add_argument("--neutral_scan_start", type=float, default=0.30)
    parser.add_argument("--neutral_scan_end", type=float, default=0.70)
    parser.add_argument("--neutral_scan_step", type=float, default=0.01)
    parser.add_argument(
        "--write_best_neutral_threshold",
        action="store_true",
        help="When scanning, write best neutral_threshold back to emotions.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "train", "all"],
        default="val",
        help="Which split to evaluate. 'val' matches modelCreation.py train_test_split(test_size, seed).",
    )
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = args.model_dir
    data_path = args.data_path
    max_length = args.max_length

    # =======================
    # Load metadata
    # =======================
    meta_path = os.path.join(model_dir, "emotions.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    emotions = meta["emotions"]
    threshold = float(meta.get("threshold", 0.5))

    # You said you already added these fields:
    neutral_threshold = float(meta.get("neutral_threshold", 0.35))

    max_labels = int(meta.get("max_labels", 3))

    if args.threshold is not None:
        threshold = float(args.threshold)
    if args.neutral_threshold is not None:
        neutral_threshold = float(args.neutral_threshold)
    if args.max_labels is not None:
        max_labels = int(args.max_labels)

    print("=== Evaluation Settings ===")
    print("model_dir:", model_dir)
    print("data_path:", data_path)
    print("split:", args.split)
    print("seed/test_size:", args.seed, "/", args.test_size)
    print("emotions:", emotions)
    print("threshold:", threshold)
    print("neutral_threshold:", neutral_threshold)
    print("max_labels:", max_labels)
    print("===========================\n")

    # =======================
    # Load dataset
    # =======================
    data = read_jsonl(data_path)

    texts = [x["text"] for x in data]
    # Expecting: obj["emotions"] is dict like {"anger":0/1, ...} possibly all zeros for neutral
    y_true = np.array(
        [[int(x.get("emotions", {}).get(e, 0)) for e in emotions] for x in data],
        dtype=int
    )

    if args.split != "all":
        # Prefer a stable split saved by training (prevents leakage arguments).
        split_path = os.path.join(model_dir, "split_indices.json")
        if os.path.exists(split_path):
            with open(split_path, "r", encoding="utf-8") as f:
                split = json.load(f)
            train_idx = split.get("train_idx", [])
            val_idx = split.get("val_idx", [])

            if args.split == "train":
                idx = train_idx
            else:
                idx = val_idx

            texts = [texts[i] for i in idx]
            y_true = np.asarray([y_true[i] for i in idx], dtype=int)
            print("Using saved split:", split_path)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                texts,
                y_true,
                test_size=args.test_size,
                random_state=args.seed,
                shuffle=True,
            )
            if args.split == "train":
                texts, y_true = X_train, np.asarray(y_train, dtype=int)
            else:
                texts, y_true = X_val, np.asarray(y_val, dtype=int)

    print(f"Evaluating on {len(texts)} samples\n")

    # =======================
    # Load model
    # =======================
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # =======================
    # Predict (batched)
    # =======================
    batch_size = 16
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().numpy()

        probs = sigmoid(logits)
        all_probs.append(probs)

    probs = np.vstack(all_probs)  # shape: [N, num_labels]

    # =======================
    # Optional: scan neutral_threshold
    # =======================
    if args.scan_neutral_threshold:
        max_probs = probs.max(axis=1)
        true_is_neutral = (y_true.sum(axis=1) == 0)

        best = None
        start = float(args.neutral_scan_start)
        end = float(args.neutral_scan_end)
        step = float(args.neutral_scan_step)
        if step <= 0:
            raise ValueError("neutral_scan_step must be > 0")

        thr = start
        while thr <= end + 1e-12:
            thr_r = round(float(thr), 2)
            pred_is_neutral = (max_probs < thr_r)

            neutral_tp = int((true_is_neutral & pred_is_neutral).sum())
            neutral_fp = int((~true_is_neutral & pred_is_neutral).sum())
            neutral_fn = int((true_is_neutral & ~pred_is_neutral).sum())

            neutral_precision = neutral_tp / (neutral_tp + neutral_fp) if (neutral_tp + neutral_fp) > 0 else 0.0
            neutral_recall = neutral_tp / (neutral_tp + neutral_fn) if (neutral_tp + neutral_fn) > 0 else 0.0
            neutral_f1 = (
                (2 * neutral_precision * neutral_recall) / (neutral_precision + neutral_recall)
                if (neutral_precision + neutral_recall) > 0
                else 0.0
            )

            cand = {
                "neutral_threshold": thr_r,
                "neutral_f1": float(neutral_f1),
                "neutral_precision": float(neutral_precision),
                "neutral_recall": float(neutral_recall),
                "neutral_tp": neutral_tp,
                "neutral_fp": neutral_fp,
                "neutral_fn": neutral_fn,
            }

            if best is None:
                best = cand
            else:
                if cand["neutral_f1"] > best["neutral_f1"]:
                    best = cand
                elif cand["neutral_f1"] == best["neutral_f1"] and cand["neutral_precision"] > best["neutral_precision"]:
                    best = cand

            thr += step

        print("\n=== Neutral Threshold Scan ===")
        print(
            f"Range: [{start:.2f}, {end:.2f}] step={step:.2f} | N={len(texts)} | true_neutral={(true_is_neutral.sum())}"
        )
        print("BEST:", best)

        if args.write_best_neutral_threshold:
            meta["neutral_threshold"] = float(best["neutral_threshold"])
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"Updated {meta_path} with neutral_threshold={best['neutral_threshold']}")

    # =======================
    # Apply your decoding logic
    # =======================
    y_pred = np.zeros_like(y_true)

    pred_none = 0
    true_none = int((y_true.sum(axis=1) == 0).sum())

    for i in range(len(texts)):
        idxs = decode_predictions(
            probs_row=probs[i],
            emotions=emotions,
            threshold=threshold,
            neutral_threshold=neutral_threshold,
            max_labels=max_labels
        )
        if len(idxs) == 0:
            pred_none += 1
        else:
            y_pred[i, idxs] = 1

    # =======================
    # Metrics
    # =======================
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Subset accuracy (exact match per sample)
    subset_acc = float((y_true == y_pred).all(axis=1).mean())

    # Neutral detection metrics (treat neutral as y.sum==0)
    true_is_neutral = (y_true.sum(axis=1) == 0)
    pred_is_neutral = (y_pred.sum(axis=1) == 0)

    neutral_tp = int((true_is_neutral & pred_is_neutral).sum())
    neutral_fp = int((~true_is_neutral & pred_is_neutral).sum())
    neutral_fn = int((true_is_neutral & ~pred_is_neutral).sum())

    neutral_precision = neutral_tp / (neutral_tp + neutral_fp) if (neutral_tp + neutral_fp) > 0 else 0.0
    neutral_recall = neutral_tp / (neutral_tp + neutral_fn) if (neutral_tp + neutral_fn) > 0 else 0.0

    # =======================
    # Print summary
    # =======================
    print("\n=== Overall Metrics ===")
    print(f"Micro  F1: {micro_f1:.4f} | P: {micro_p:.4f} | R: {micro_r:.4f}")
    print(f"Macro  F1: {macro_f1:.4f} | P: {macro_p:.4f} | R: {macro_r:.4f}")
    print(f"Subset Accuracy (Exact match): {subset_acc:.4f}")

    print("\n=== Neutral Diagnostics ===")
    print(f"True neutral count: {true_none} / {len(texts)} ({true_none/len(texts):.2%})")
    print(f"Pred neutral count: {pred_none} / {len(texts)} ({pred_none/len(texts):.2%})")
    print(f"Neutral Precision: {neutral_precision:.4f}")
    print(f"Neutral Recall:    {neutral_recall:.4f}")
    print(f"Neutral TP/FP/FN:  {neutral_tp}/{neutral_fp}/{neutral_fn}")

    print("\n=== Per-label Report ===")
    print(classification_report(y_true, y_pred, target_names=emotions, zero_division=0))

    # Optional: show a few worst-looking examples (high confidence wrong / weird)
    # Here: samples where model predicted neutral but true not neutral, or vice versa
    print("\n=== Example Errors (up to 10) ===")
    shown = 0
    for i in range(len(texts)):
        if shown >= 10:
            break

        t_neu = bool(true_is_neutral[i])
        p_neu = bool(pred_is_neutral[i])

        if t_neu != p_neu:
            true_labels = [emotions[j] for j in range(len(emotions)) if y_true[i, j] == 1] or ["none"]
            pred_labels = [emotions[j] for j in range(len(emotions)) if y_pred[i, j] == 1] or ["none"]

            print("\nTEXT:", texts[i])
            print("TRUE:", true_labels)
            print("PRED:", pred_labels)
            print("PROBS:", {emotions[j]: float(probs[i, j]) for j in range(len(emotions))})

            shown += 1


if __name__ == "__main__":
    main()
