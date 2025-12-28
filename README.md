# Clinical EmotiSupport 

Multi-label emotion classification for patient/caregiver messages in a telemedicine context. The project includes synthetic data generation, data cleaning + consistency enforcement, EDA, a classical baseline model, and a fine-tuned DistilBERT model.

## Journey (from interim feedback to the final baseline)

After the interim feedback, we moved from a weaker/less-defensible approach to a clear, measurable, and reproducible pipeline:

1. **Task definition** – instead of regression / a single "emotion score", we defined **multi-label classification**: each message can have 0..N emotions active simultaneously.
2. **Synthetic data generation** – generated realistic messages for two domains (clinical/administrative) using Ollama.
3. **Quality checks + EDA** – checked duplicates, length/domain distributions, label supports, and obvious contradictions.
4. **Data Quality Policy (consistency rule)** – if an explicit emotion word appears in the text (e.g., “angry”, “confused”), the corresponding label must be 1.
5. **Classical baseline** – TF‑IDF + Logistic Regression (OvR) for a fast, interpretable reference point.
6. **DistilBERT model** – fine-tuned a multi-label model with BCEWithLogitsLoss.
7. **Reproducibility without leakage** – saved train/val split indices so all models are evaluated on the same split.
8. **Targeted improvement** – strengthened anger-focused examples after observing weaker performance for anger.

## Labels

The project classifies the following 6 emotions:

- anxiety
- confusion
- frustration
- anger
- disappointment
- satisfaction

## Repository layout (Git root = `Code/`)

- `Data/` – data, generation, cleaning, and EDA (JSONL)
  - `dataset.jsonl` – raw/augmented dataset
  - `dataset.cleaned.jsonl` – cleaned dataset (policy-consistent labels)
  - `dataGeneration.py` – synthetic data generation (Ollama)
  - `dataClean.py` – cleaning: enforce “explicit emotion word ⇒ label=1”
  - `dataEDA.py` – EDA (writes `eda_outputs*/eda_report.json`)
- `Model/` – training/evaluation/baseline
  - `modelCreation.py` – DistilBERT fine-tuning (multi-label)
  - `modelEvaluation.py` – evaluation + decoding (threshold + neutral gate)
  - `baselineTfidfLogReg.py` – TF‑IDF + Logistic Regression baseline
- `.github/` – repo instructions and workflow rules

## Presentation files

The presentation files are located in the repo root (`Code/`):

- `Clinical EmotiSupport - Interim Presentation.pptx`
- `Clinical EmotiSupport - Interim Presentation.pdf`

## Data format (JSONL)

Each line is a JSON object with these main fields:

- `text`: the message text
- `domain`: clinical / administrative
- `language`: en
- `emotions`: an object with emotion keys and 0/1 values

## Results (latest runs)

Dataset (cleaned):

- Size: 1132 messages
- Domain split: clinical 564 (49.8%), administrative 568 (50.2%)
- No exact-text duplicates
- 0 “explicit emotion word present but label=0” mismatches (per EDA lexicon)

Baseline (TF‑IDF + Logistic Regression), aligned split (Train 962 / Val 170):

- Micro F1: 0.7787
- Macro F1: 0.7741

DistilBERT (v5 anger‑focused run), evaluated on the same val split:

- Micro F1: 0.7864
- Macro F1: 0.7971

## Reproducibility / Train‑Val split

To avoid data leakage and ensure a fair comparison between models:

- The DistilBERT run stores `split_indices.json` inside the model folder.
- The classical baseline can align to the same split via `--align_with_model_dir`.

## Quick run (CLI)

EDA on the cleaned dataset:

```powershell
python "Data\dataEDA.py" --data_path "Data\dataset.cleaned.jsonl" --save_dir "Data\eda_outputs_cleaned" --save_plots --no_show
```

DistilBERT evaluation (v5):

```powershell
python "Model\modelEvaluation.py" --model_dir "..\Baseline Model\Model\distilbert_run_v5_anger" --data_path "Data\dataset.cleaned.jsonl" --split val
```

TF‑IDF baseline (aligned to v5 split):

```powershell
python "Model\baselineTfidfLogReg.py" --data_path "Data\dataset.cleaned.jsonl" --align_with_model_dir "..\Baseline Model\Model\distilbert_run_v5_anger" --threshold 0.5
```
