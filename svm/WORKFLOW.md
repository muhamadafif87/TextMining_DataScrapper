# SVM Emotion Model Workflow

This file is the progress record for building and validating the SVM model.

## Current status

**Current phase: Ekman-7 SVM training and held-out test evaluation complete.**

| Phase | Status | Exit condition |
|---|---|---|
| Dataset inspection | Complete | Source and dataset quality limits documented |
| Rule-based label proposals | Complete | Review CSV generated from the source dataset |
| LLM label validation (`Ekman-7`) | Complete | All 4,425 valid rows independently labeled in `manual_label` without reading auto-labels; 380 `ambigu` rows flagged in `manual_notes` |
| Group-aware data split | Complete (`output/dataset_split.csv`) | 50.00% train (2,192) / 20.03% val (878) / 29.97% test (1,314) across 178 conversation groups with 0 group overlap |
| SVM training and tuning | Complete (`svm/train_svm.py`) | Best validation config: `LinearSVC_C0.3_cw-None_ng12` (`val_macro_f1=0.2921`, `val_acc=0.5524`) |
| Final evaluation | Complete (`output/svm_ekman7_evaluation_report.json`) | Held-out test (1,314 rows): **Accuracy = 0.5936**, **Macro-F1 = 0.3375**, **Weighted-F1 = 0.5261** |

## Dataset and review files

- Source: `../cleaned_data.csv` (4,633 rows; `sentiment_label` was empty at inspection).
- Review sheet: `../output/cleaned_data_auto_label_review.csv`.
- The review sheet contains proposed 9-class, Ekman-7, and main-5 labels, scores, heuristic confidence, and matched cues.
- `manual_label` is intentionally blank. Choose one scheme before entering labels; do not combine labels from different schemes in that column.
- `is_valid_opinion` and `filter_reason` identify rows eligible for model training. Excluded rows have no proposed emotion labels.
- Keep the original `sentiment_label` untouched. Auto-labels are suggestions, not ground truth, and `auto_confidence_heuristic` is not a calibrated probability.
- For an unbiased evaluation, assign `manual_label` without looking at the auto-label columns, at least for the final test data. Auto-labels can be compared afterward or used as suggestions for training annotations, but copying them into test labels would make the SVM evaluation circular.

## Agreed evaluation split

The requested proportions are **50% train / 20% validation / 30% test**. Apply the split after manual review and duplicate handling. Keep each `conversation_id` (or its `root_tweet_id`) in only one split to prevent replies from the same conversation leaking across sets. Aim for class balance while respecting those groups; report actual counts if exact percentages cannot be achieved.

## Next actions

1. Select the target label scheme: 9-class, Ekman-7, or main-5.
2. Review each eligible row and fill `manual_label` with the confirmed/corrected label; use `manual_notes` for ambiguous cases.
3. Check missing labels, class counts, duplicate text, and conversation-group overlap before splitting.
4. Train and tune only after the manual labels and split are frozen.
