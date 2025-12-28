# Clinical EmotiSupport (Interim)

Multi-label emotion classification for patient/caregiver messages (telemedicine context). Includes data generation, EDA, classical baseline, and a DistilBERT fine-tuned model.

## Repository layout (Git root = `Code/`)

- `Data/` – data generation + EDA + datasets (JSONL)
  - `dataset.jsonl` – raw/augmented dataset
  - `dataset.cleaned.jsonl` – cleaned dataset (policy-consistent labels)
  - `dataGeneration.py` – synthetic data generation (Ollama)
  - `dataClean.py` – enforce policy: explicit emotion word ⇒ label=1
  - `dataEDA.py` – EDA runner (writes `eda_outputs*/eda_report.json` + optional PNGs)
- `Model/` – training/evaluation scripts
  - `modelCreation.py` – DistilBERT fine-tuning (multi-label)
  - `modelEvaluation.py` – evaluation + decoding (threshold + neutral gate)
  - `baselineTfidfLogReg.py` – TF-IDF + Logistic Regression baseline
- `notebooks/` – submission notebooks (generation / EDA / baseline evaluation)
- `notebooks_py/` – Python equivalents (source-of-truth if you prefer `.py`)
- `.github/` – repo instructions (`copilot-instructions.md`)

## Presentation files

Place the interim presentation files in the Git root (`Code/`):
- `Clinical EmotiSupport - Interim Presentation.pptx`
- `Clinical EmotiSupport - Interim Presentation.pdf`

## Quick run (CLI)

EDA:
```powershell
python "Data\dataEDA.py" --data_path "Data\dataset.cleaned.jsonl" --save_dir "Data\eda_outputs_cleaned" --save_plots --no_show
```

DistilBERT evaluation (v5):
```powershell
python "Model\modelEvaluation.py" --model_dir "..\Baseline Model\Model\distilbert_run_v5_anger" --data_path "Data\dataset.cleaned.jsonl" --split val
```

TF-IDF baseline (aligned to v5 split):
```powershell
python "Model\baselineTfidfLogReg.py" --data_path "Data\dataset.cleaned.jsonl" --align_with_model_dir "..\Baseline Model\Model\distilbert_run_v5_anger" --threshold 0.5
```
