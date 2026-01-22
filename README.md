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

| Label | Count | % of Data | Note |
| :--- | :--- | :--- | :--- |
| **Anxiety** | 996 | 49.9% | Dominant emotion (High baseline) |
| **Frustration** | 957 | 47.9% | Key indicator for churn risk |
| **Confusion** | 950 | 47.6% | Indicates UX/Process failures |
| **Disappointment** | 695 | 34.8% | Service gap indicator |
| **Anger** | 628 | 31.5% | Moderately imbalanced (Require low threshold) |
| **Satisfaction** | 110 | 5.5% | **Highly Imbalanced** (Requires high precision threshold) |

## Test Set Metrics (Detailed)

Evaluated on N=400 split using Business-Aligned thresholds.

**Overall:**
- **Micro F1:** 0.84
- **Macro F1:** 0.80

**Per-Label Performance:**

| Emotion | Precision | Recall | F1 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Anxiety** | 0.97 | 0.87 | **0.92** | Excellent reliability |
| **Confusion** | 0.92 | 0.85 | **0.89** | Strong operational signal |
| **Frustration** | 0.95 | 0.82 | **0.88** | Reliable churn prediction |
| **Disappointment** | 0.87 | 0.75 | **0.81** | Good, but subjective |
| **Satisfaction** | 0.73 | 0.67 | **0.70** | Challenged by low sample size (5.5%) |
| **Anger** | 0.66 | 0.59 | **0.62** | Trade-off: Lower precision accepted for higher recall |

**Neutral Detection:**
- **Precision:** 0.94 (Trustworthy "noise filter")

## Limitations & Failure Analysis

While the model performs well (0.84 Micro F1), users should be aware of known limitations:

1.  **Politeness vs. Sentiment Gap:** The model sometimes struggles with "Polite Anger" — highly formal messages that convey fury without using explicit words. We addressed this with lower Anger thresholds, but some subtle sarcasm may still be missed.
2.  **Short-Text Ambiguity:** Messages with <5 words (e.g., "Not yet", "Waiting") lack context and may default to Neutral even if the user is annoyed.
3.  **Satisfaction Scarcity:** With only 5.5% satisfaction data, the model is conservative about predicting positive outcomes. It requires strong evidence ("perfect", "thank you so much") to trigger this label.

## Training Environment

- **Hardware:** Single NVIDIA GPU (Consumer Grade) / Standard CPU.
- **Training Time:** ~90 minutes (5 epochs).
- **Inference Latency:** <100ms per message (Batch size 1, CPU).

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
