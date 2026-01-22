# Clinical EmotiSupport

Multi-label emotion classification for patient/caregiver messages in a telemedicine context. The project includes synthetic data generation, rigorous data cleaning, consistency enforcement, EDA, and a fine-tuned Bio-ClinicalBERT model with **business-aligned inference logic**.

Model in Hugging Face:
```https://huggingface.co/YuvalRez1/clinical-emotisupport-bert```

## Journey (from interim feedback to the final v2 model)

After the interim feedback, we moved from a weaker approach to a robust, reproducible, and product-ready pipeline:

1. **Task definition** – Multi-label classification: each message can have 0..N emotions active simultaneously.
2. **Synthetic data generation** – Generated 2,000 realistic messages (clinical/administrative) using Ollama (DeepSeek-R1).
3. **Aggressive Data Cleaning (v6)** – Removed logical contradictions (e.g., "polite anger" where patients say "thank you" while furious) and stripped artificial templates.
4. **Data Quality Policy** – Enforced: "Explicit emotion word ⇒ Label=1" and "No negative emotion + Satisfaction allowed simultaneously".
5. **Business-Aligned Logic** – Implemented **Adaptive Thresholding** (low bar for anger/risk, high bar for satisfaction) and a **Neutral Guardrail** to prevent false alarms.
6. **Bio-ClinicalBERT** – Fine-tuned `emilyalsentzer/Bio_ClinicalBERT` for domain-specific language understanding.

## Labels

The project classifies the following 6 emotions:

- **Anxiety** (Threshold: 0.40) - High sensitivity for early distress detection.
- **Confusion** (Threshold: 0.50) - Balanced.
- **Frustration** (Threshold: 0.40) - High sensitivity for churn prevention.
- **Anger** (Threshold: 0.30) - **Critical sensitivity**: better to flag false alarm than miss a furious patient.
- **Disappointment** (Threshold: 0.50) - Balanced.
- **Satisfaction** (Threshold: 0.65) - High precision required (must be truly satisfied, not just polite).

## Repository layout

- `Data/`
  - `dataset_refined_v6_reindexed.jsonl` – The final, cleaned, re-indexed dataset (1,996 records).
  - `emotions.json` – Configuration file defining labels and **business logic thresholds**.
  - `dataGeneration.py` / `dataAugment.py` – Synthetic generation pipeline.
  - `refine_labels.py` – The script used to clean "politeness artifacts" and fix logic conflicts.
- `Model/`
  - `modelCreation.py` – Training script (saves `split_indices.json` for reproducibility).
  - `modelEvaluation.py` – Evaluation script with **Guardrail & Adaptive Threshold logic**.
- `.github/` – Copilot instructions.

## Results (Final v2 Run)

**Dataset (v6 refined):**
- Size: 1,996 purified messages.
- 0 Logical Conflicts (e.g., no Satisfaction + Anger).
- 0 Text Artifacts (cleaned headers/templates).

**Test Set Metrics (N=400, 20% split):**

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Micro F1** | **0.84** | Strong overall performance |
| **Neutral Precision** | **0.94** | Highly trustworthy when filtering noise |
| **Anger Recall** | **0.59** | Massive improvement (vs ~0.45 in baseline) due to adaptive thresholding |
| **Subset Accuracy** | **0.55** | Exact match ratio |

## Reproducibility

To ensure fair comparison, the training process saves `split_indices.json`.
The evaluation script automatically loads these indices if available.

## Quick run (CLI)

1. **EDA on the final dataset:**
```powershell
python "Data\dataEDA.py" --data_path "Data\dataset_refined_v6_reindexed.jsonl"
```

2. **Train the Model (creates ClinicalBERT_Run_v1):**
```powershell
python "Model\modelCreation.py" --data_path "Data\dataset_refined_v6_reindexed.jsonl" --out_dir "ClinicalBERT_Run_v1" --epochs 5 --batch 8 --lr 2e-5
```

3. **Evaluate with Business Logic (Thresholds + Guardrail):**
*Note: This script automatically loads thresholds from `Data/emotions.json`*
```powershell
python "Model\modelEvaluation.py" --model_dir "ClinicalBERT_Run_v1" --data_path "Data\dataset_refined_v6_reindexed.jsonl" --split val
```
