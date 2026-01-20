import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


EMOTIONS = ["anxiety", "confusion", "frustration", "anger", "disappointment", "satisfaction"]


def read_jsonl(path: str) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    labels: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
            emo = obj.get("emotions", {}) or {}
            labels.append([int(emo.get(e, 0) or 0) for e in EMOTIONS])
    return texts, np.asarray(labels, dtype=int)


def load_split_indices(model_dir: str):
    split_path = os.path.join(model_dir, "split_indices.json")
    if not os.path.exists(split_path):
        return None
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_split(texts: List[str], y: np.ndarray, split: dict, which: str):
    train_idx = split.get("train_idx", [])
    val_idx = split.get("val_idx", [])
    if which == "train":
        idx = train_idx
    elif which == "val":
        idx = val_idx
    else:
        raise ValueError("which must be train|val")

    if not idx:
        raise ValueError("Split indices are empty")
    if max(idx) >= len(texts):
        raise ValueError(
            f"Split indices out of range for current dataset length={len(texts)}. "
            "Dataset was probably edited/reordered after training."
        )

    X = [texts[i] for i in idx]
    Y = np.asarray([y[i] for i in idx], dtype=int)
    return X, Y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline for multi-label emotions")
    parser.add_argument("--data_path", type=str, default=r"Data\dataset.jsonl")
    parser.add_argument(
        "--align_with_model_dir",
        type=str,
        default=None,
        help="If set, use its split_indices.json for a fair comparison to DistilBERT.",
    )
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    texts, y = read_jsonl(args.data_path)

    split = None
    if args.align_with_model_dir:
        split = load_split_indices(args.align_with_model_dir)
        if split:
            print("Using saved split:", os.path.join(args.align_with_model_dir, "split_indices.json"))

    if split:
        X_train, y_train = apply_split(texts, y, split, "train")
        X_val, y_val = apply_split(texts, y, split, "val")
    else:
        idx = list(range(len(texts)))
        train_idx, val_idx = train_test_split(idx, test_size=args.test_size, random_state=args.seed, shuffle=True)
        X_train = [texts[i] for i in train_idx]
        y_train = np.asarray([y[i] for i in train_idx], dtype=int)
        X_val = [texts[i] for i in val_idx]
        y_val = np.asarray([y[i] for i in val_idx], dtype=int)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, int(args.ngram_max)),
        min_df=int(args.min_df),
        max_features=int(args.max_features),
    )

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=3000,
            C=float(args.C),
            solver="liblinear",
            class_weight="balanced",
        )
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf.fit(X_train_vec, y_train)

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_val_vec)
        probs = sigmoid(np.asarray(scores))
    else:
        probs = clf.predict_proba(X_val_vec)

    thr = float(args.threshold)
    y_pred = (probs >= thr).astype(int)

    micro_f1 = f1_score(y_val, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    micro_p = precision_score(y_val, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_val, y_pred, average="micro", zero_division=0)

    print("=== TF-IDF + Logistic Regression (OvR) Baseline ===")
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
    print(f"threshold: {thr}")
    print(f"Micro  F1: {micro_f1:.4f} | P: {micro_p:.4f} | R: {micro_r:.4f}")
    print(f"Macro  F1: {macro_f1:.4f}")
    print("\n=== Per-label Report ===")
    print(classification_report(y_val, y_pred, target_names=EMOTIONS, zero_division=0))


if __name__ == "__main__":
    main()
