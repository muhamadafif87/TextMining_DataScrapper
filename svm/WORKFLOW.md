# SVM Emotion Model Workflow

This file is the progress record for building and validating the SVM model.

## Current status

**Current phase: human review of proposed labels.** No SVM training or model evaluation has been completed.

| Phase | Status | Exit condition |
|---|---|---|
| Dataset inspection | Complete | Source and dataset quality limits documented |
| Rule-based label proposals | Complete | Review CSV generated from the source dataset |
| Human label validation | In progress | Select one label scheme, then enter independent human labels for usable rows in `manual_label` |
| Group-aware data split | Not started | Freeze train/validation/test assignment with no conversation group crossing splits |
| SVM training and tuning | Not started | Train only on manual labels; select settings using validation data |
| Final evaluation | Not started | Evaluate once on the held-out test set; report macro-F1 and per-class metrics |

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
