<!-- .github/copilot-instructions.md - guidance for AI coding agents -->
# Copilot / AI Agent Instructions for this repo

Purpose: quickly orient an AI editing assistant to the project's architecture, important files, developer workflows, and project-specific conventions so changes are correct and low-risk.

1) Big picture
- This repo produces a multi-label emotion classifier for patient/caregiver messages. Data is JSONL (one JSON object per line). The pipeline splits into two main areas:
  - Data generation & EDA: `Data/*` — synthetic data generation uses an Ollama model (`dataGeneration.py`) and analysis/EDA lives in `dataEDA.py`.
  - Modeling & evaluation: `Model/*` — training (`modelCreation.py`), threshold tuning (`tuneThreshhold.py`), evaluation (`modelEvaluation.py`) and lightweight prediction demo (`predictDemo.py`).

2) Data format and conventions (critical)
- Each dataset line is a JSON object with at least: `text`, `domain`, `language`, `emotions` (dict of 0/1 ints). Example:
  {"text":"...","domain":"clinical","language":"en","emotions":{"anxiety":1,...}}
- Labels are multi-hot (0/1). Some scripts include an `id` field for generated data.
- File storage: many scripts reference `..\Baseline Model\Model\distilbert_run_*` — run scripts from the repository root (`Code/`) so relative paths resolve.

3) Important project-specific patterns
- Multi-label classification: training uses `BCEWithLogitsLoss` with `pos_weight` computed per-label in `modelCreation.py` (see `compute_pos_weight`). If you change label set, keep this in sync.
- Custom Trainer: `WeightedTrainer` overrides `compute_loss` to apply BCEWithLogitsLoss with `pos_weight`.
- Emotions label list: keep `EMOTIONS` consistent across `Model/*` and EDA/cleaning code.
- Saved model metadata: `modelCreation.py` writes `emotions.json` with `emotions` + decoding params. Consumers expect:
  `{"emotions": [...], "threshold": 0.5, "neutral_threshold": 0.35, "max_labels": 3}`

4) External dependencies & infra
- Ollama: `Data/dataGeneration.py` uses `ollama.chat`. Ensure `ollama` is running and the model name (e.g., `deepseek-r1:8b`) is available.
- Transformers / Hugging Face: training and evaluation use `transformers`, `datasets`, `sklearn`, and `torch`. Evaluation loads local model artifacts with `local_files_only=True`.

5) Developer workflows & commands (examples)
- Generate synthetic data (requires Ollama + model):
  ```powershell
  python "Data\dataGeneration.py"
  ```
- Train model:
  ```powershell
  python "Model\modelCreation.py" --data_path "Data\dataset.cleaned.jsonl" --out_dir "..\Baseline Model\Model\distilbert_run_new" --model_name distilbert-base-uncased
  ```
- Evaluate:
  ```powershell
  python "Model\modelEvaluation.py" --model_dir "..\Baseline Model\Model\distilbert_run_v5_anger" --data_path "Data\dataset.cleaned.jsonl" --split val
  ```

6) Code-editing guidance for AI agents (do these)
- Preserve label ordering and shape: all scripts assume consistent `EMOTIONS` ordering.
- Prefer path-safety: accept CLI args and/or use `os.path.abspath()`.
- Keep Windows encoding in mind: avoid breaking `sys.stdout.reconfigure(encoding="utf-8")` patterns.
