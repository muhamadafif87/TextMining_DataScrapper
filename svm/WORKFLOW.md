# SVM 3-Class Sentiment Model Workflow

This document records the roadmap, architecture, and empirical progress for building and validating the SVM sentiment classification pipeline for public opinion on Perpres No. 111 Tahun 2025.

---

## 1. Project Pivot & Context

The project has transitioned from fine-grained emotion classification (**Ekman-7**: *marah, netral, sedih, senang, terkejut, takut, jijik*) to a streamlined **3-class sentiment classification** (**positif, netral, negatif**, with `noise` filtered out).

- **Prior State (Ekman-7 / Track B):** Trained on `output/cleaned_data_auto_label_review.csv` (`manual_label`). While functional, 7-class emotion suffered from severe class imbalance (*marah* ~52.4%, *takut*/*jijik*/*senang* minority with low recall), resulting in held-out test Macro-F1 = 0.3375 and Accuracy = 0.5936.
- **Current State (3-Class Sentiment / Track A):** Uses `cleaned_data.csv` (`sentiment_label`), which has been fully labeled (4,633 rows) through a 3-pass stance-focused adjudication taking `root_tweet_text` into account.

---

## 2. Dataset Overview (`cleaned_data.csv`)

| Sentiment Label | Total Rows | Proportion (Total) | Status in Modeling | Description |
|---|---:|---:|---|---|
| `netral` | 2,282 | 49.26% | **Target Class** | Factual news, objective queries, or discussion outside policy stance |
| `negatif` | 1,533 | 33.09% | **Target Class** | Criticism, rejection of Perpres 111/2025 (human rights, discrimination, distraction claims) |
| `positif` | 553 | 11.94% | **Target Class** | Support, approval, or defense of Perpres 111/2025 & government policy |
| `noise` | 265 | 5.72% | **Excluded (Filter)** | Mentions/URLs only, spam, gambling, promo, or uncontextual fragments |
| **Total** | **4,633** | **100.00%** | **4,368 Valid Rows** | Spans 177 distinct `conversation_id` groups |

After text preprocessing (normalizing informal text, removing URLs/mentions, negation binding, emoji token mapping), 9 rows became empty and 22 exact duplicate texts were removed, leaving **4,337 usable rows** across **177 unique conversation groups**.

---

## 3. Workflow Phases & Current Status

**Current phase: 3-Class Sentiment SVM training and held-out test evaluation COMPLETE.**

| Phase | Status | Exit Condition / Deliverable |
|---|---|---|
| **Phase 1: Dataset & Label Verification** | **Complete** | All 4,633 rows labeled in `cleaned_data.csv`; 265 `noise` rows tagged and excluded. |
| **Phase 2: Text Preprocessing & Cleaning** | **Complete** | `svm/preprocessing.py` normalization, negation joining, and duplicate removal verified. |
| **Phase 3: Group-Aware Stratified Split** | **Complete (`output/dataset_split_sentiment.csv`)** | 50.01% train (2,169 rows, 104 groups) / 19.99% val (867 rows, 25 groups) / 30.00% test (1,301 rows, 48 groups) across 177 groups with **0 thread leakage**. |
| **Phase 4: SVM Pipeline & Hyperparameter Tuning** | **Complete (`output/svm_sentiment3_tuning_results.csv`)** | 20 hyperparameter candidates tuned on validation set. Best model: `SVC_RBF_C1.0_cw-balanced_ng12` (`val_macro_f1=0.4985`, `val_weighted_f1=0.5810`, `val_acc=0.5998`). |
| **Phase 5: Held-Out Test Evaluation & Export** | **Complete (`output/svm_sentiment3_evaluation_report.json`)** | Single-run test set evaluation (1,301 rows): **Accuracy = 0.6341**, **Macro-F1 = 0.5752**, **Weighted-F1 = 0.6187**. |
| **Phase 6: Predictor & Dashboard Integration** | **Complete (`svm/predictor.py`)** | `SentimentPredictor` class ready with calibrated probability outputs, UI styles (`#2CA02C`, `#7F7F7F`, `#D62728`), and CLI demo. |

---

## 4. Empirical Evaluation Results (Held-Out Test Set)

Evaluated on **1,301 unseen rows** from 48 distinct conversation groups:

- **Overall Accuracy:** `0.6341` (63.41%)
- **Macro-F1:** `0.5752` (57.52% — significant improvement over Ekman-7's 0.3375)
- **Weighted-F1:** `0.6187` (61.87%)

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support (Rows) |
|---|---:|---:|---:|---:|
| `positif` | **0.7262** | 0.3720 | 0.4919 | 164 |
| `netral` | 0.6435 | **0.8189** | **0.7207** | 679 |
| `negatif` | 0.5892 | 0.4541 | 0.5129 | 458 |
| **Macro Avg** | **0.6530** | **0.5483** | **0.5752** | 1,301 |
| **Weighted Avg** | **0.6348** | **0.6341** | **0.6187** | 1,301 |

---

## 5. Generated Artifacts & File Locations

| File / Folder | Role & Content |
|---|---|
| [cleaned_data.csv](file:///d:/Coolyeah/DatMin/TBP/cleaned_data.csv) | Source dataset with `sentiment_label` (4,633 rows) |
| [svm/train_svm.py](file:///d:/Coolyeah/DatMin/TBP/svm/train_svm.py) | Training & tuning script with 3-class sentiment support and group-aware split |
| [svm/predictor.py](file:///d:/Coolyeah/DatMin/TBP/svm/predictor.py) | Dashboard-ready inference API (`SentimentPredictor`, `predict_sentiment`) |
| [output/dataset_split_sentiment.csv](file:///d:/Coolyeah/DatMin/TBP/output/dataset_split_sentiment.csv) | Frozen group-aware 50/20/30 split mapping |
| [output/svm_sentiment3_tuning_results.csv](file:///d:/Coolyeah/DatMin/TBP/output/svm_sentiment3_tuning_results.csv) | Tuning table of 20 candidate SVM configs on validation set |
| [output/svm_sentiment3_evaluation_report.json](file:///d:/Coolyeah/DatMin/TBP/output/svm_sentiment3_evaluation_report.json) | Full evaluation report payload and per-class metrics |
| [output/svm_sentiment3_confusion_matrix.png](file:///d:/Coolyeah/DatMin/TBP/output/svm_sentiment3_confusion_matrix.png) | Heatmap visualization of test set confusion matrix |
| [output/svm_sentiment3_confusion_matrix.csv](file:///d:/Coolyeah/DatMin/TBP/output/svm_sentiment3_confusion_matrix.csv) | Tabular confusion matrix values |
| [models/svm_sentiment_pipeline.joblib](file:///d:/Coolyeah/DatMin/TBP/models/svm_sentiment_pipeline.joblib) | Serialized best sentiment model pipeline (also linked as `best_svm_pipeline.joblib`) |
| [models/model_metadata.json](file:///d:/Coolyeah/DatMin/TBP/models/model_metadata.json) | Metadata containing best config and test metrics |

---

## 6. How to Run & Verify

1. **Retrain or Tune Models:**
   ```bash
   .\.venv\Scripts\python.exe -m svm.train_svm --scheme sentiment3
   ```
2. **Run One-Shot Inference via CLI:**
   ```bash
   .\.venv\Scripts\python.exe -m svm.predictor "Mendukung penuh kebijakan penertiban Perpres 111 demi moral bangsa"
   ```
3. **Run Interactive / Demo Mode:**
   ```bash
   .\.venv\Scripts\python.exe -m svm.predictor
   ```
