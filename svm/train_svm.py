"""Group-aware SVM sentiment and emotion classification training and evaluation pipeline.

Follows svm/WORKFLOW.md:
- Supports 3-class sentiment analysis ('positif', 'netral', 'negatif') on `cleaned_data.csv`
  with automatic exclusion of rows labeled 'noise'.
- Also maintains backward compatibility with Ekman-7 and Main-5 emotion schemes on
  `output/cleaned_data_auto_label_review.csv`.
- Deduplicates identical preprocessed texts before splitting.
- Performs group-aware stratified 50% train / 20% validation / 30% test splitting
  by `conversation_id` (falling back to `root_tweet_id` or `tweet_id`) to prevent
  conversation leakage across splits.
- Tunes SVM hyperparameters strictly on the validation set using Macro-F1.
- Evaluates once on the held-out test set and exports artifacts compatible with
  `svm.predictor.SentimentPredictor` and `EmotionPredictor`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC, SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from svm.auto_labeler import CLASSES_9, CLASSES_EKMAN7, CLASSES_MAIN5
from svm.predictor import DEFAULT_UI_STYLE, SCHEME_FILENAMES
from svm.preprocessing import TextCleanerTransformer, preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SENTIMENT_INPUT = PROJECT_ROOT / "cleaned_data.csv"
DEFAULT_EMOTION_INPUT = PROJECT_ROOT / "output" / "cleaned_data_auto_label_review.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SENTIMENT_SPLIT = DEFAULT_OUTPUT_DIR / "dataset_split_sentiment.csv"
DEFAULT_EMOTION_SPLIT = DEFAULT_OUTPUT_DIR / "dataset_split.csv"

CLASSES_SENTIMENT3 = ["positif", "netral", "negatif"]

SCHEME_CLASSES: dict[str, list[str]] = {
    "sentiment3": list(CLASSES_SENTIMENT3),
    "ekman7": list(CLASSES_EKMAN7),
    "main5": list(CLASSES_MAIN5),
    "9class": list(CLASSES_9),
}

EKMAN7_TO_MAIN5: dict[str, str] = {
    "marah": "marah",
    "jijik": "marah",
    "sedih": "sedih",
    "takut": "takut",
    "senang": "senang",
    "terkejut": "netral",
    "netral": "netral",
}


def _is_truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_labeled_dataset(
    input_path: Path,
    scheme: str = "sentiment3",
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load validated labels, filter noise / invalid rows, verify integrity, and deduplicate."""
    if scheme not in SCHEME_CLASSES:
        raise ValueError(f"Unsupported scheme '{scheme}'. Choose from {list(SCHEME_CLASSES)}")

    df = pd.read_csv(
        input_path,
        dtype={
            "tweet_id": "string",
            "conversation_id": "string",
            "root_tweet_id": "string",
            "in_reply_to_tweet_id": "string",
            "sentiment_label": "string",
            "manual_label": "string",
        },
        low_memory=False,
    )

    total_rows = len(df)

    if scheme == "sentiment3":
        if "raw_text" not in df.columns or "sentiment_label" not in df.columns:
            raise ValueError(
                f"Missing required columns ('raw_text', 'sentiment_label') in {input_path}"
            )

        cleaned_lbl = df["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
        noise_mask = cleaned_lbl == "noise"
        noise_count = int(noise_mask.sum())
        empty_lbl_mask = cleaned_lbl == ""
        empty_lbl_count = int(empty_lbl_mask.sum())

        valid_mask = (~noise_mask) & (~empty_lbl_mask)
        valid_df = df.loc[valid_mask].copy()
        valid_rows = len(valid_df)

        valid_df["target_label"] = cleaned_lbl.loc[valid_mask]
        allowed = set(CLASSES_SENTIMENT3)
        observed = set(valid_df["target_label"].unique())
        invalid_labels = observed - allowed
        if invalid_labels:
            raise ValueError(
                f"Invalid sentiment labels found: {sorted(invalid_labels)}. "
                f"Expected subset of {sorted(allowed)}."
            )
    else:
        required_cols = {"tweet_id", "raw_text", "is_valid_opinion", "manual_label"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in {input_path}: {sorted(missing_cols)}")

        valid_mask = df["is_valid_opinion"].map(_is_truthy)
        valid_df = df.loc[valid_mask].copy()
        valid_rows = len(valid_df)

        valid_df["manual_label"] = (
            valid_df["manual_label"].fillna("").astype(str).str.strip().str.lower()
        )
        unlabeled_count = int((valid_df["manual_label"] == "").sum())
        if unlabeled_count > 0:
            raise ValueError(
                f"Found {unlabeled_count} valid opinion rows with empty `manual_label`. "
                "Complete manual labeling before training."
            )

        allowed = set(SCHEME_CLASSES[scheme])
        observed = set(valid_df["manual_label"].unique())
        if scheme == "main5" and observed.issubset(set(CLASSES_EKMAN7)):
            valid_df["target_label"] = valid_df["manual_label"].map(EKMAN7_TO_MAIN5)
        else:
            invalid_labels = observed - allowed
            if invalid_labels:
                raise ValueError(
                    f"Invalid labels for scheme '{scheme}': {sorted(invalid_labels)}. "
                    f"Expected subset of {sorted(allowed)}."
                )
            valid_df["target_label"] = valid_df["manual_label"]

    # Build conversation group key to prevent thread leakage across splits
    conv_id = (
        valid_df.get("conversation_id", pd.Series("", index=valid_df.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    root_id = (
        valid_df.get("root_tweet_id", pd.Series("", index=valid_df.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    tweet_id = valid_df["tweet_id"].fillna("").astype(str).str.strip()
    valid_df["group_id"] = conv_id.where(conv_id != "", root_id).where(
        (conv_id != "") | (root_id != ""), tweet_id
    )

    valid_df["raw_text"] = valid_df["raw_text"].fillna("").astype(str)
    valid_df["cleaned_text"] = valid_df["raw_text"].map(preprocess_text)

    empty_cleaned = int((valid_df["cleaned_text"] == "").sum())
    if empty_cleaned > 0:
        valid_df = valid_df.loc[valid_df["cleaned_text"] != ""].copy()

    duplicates_removed = 0
    if deduplicate:
        before_dedup = len(valid_df)
        valid_df = valid_df.drop_duplicates(subset=["cleaned_text"], keep="first").copy()
        duplicates_removed = before_dedup - len(valid_df)

    valid_df = valid_df.reset_index(drop=True)
    stats: dict[str, int] = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "empty_after_preprocessing": empty_cleaned,
        "duplicates_removed": duplicates_removed,
        "usable_rows": len(valid_df),
        "unique_groups": int(valid_df["group_id"].nunique()),
    }
    if scheme == "sentiment3":
        stats["noise_rows_excluded"] = noise_count
        stats["empty_labels_excluded"] = empty_lbl_count
    return valid_df, stats


def create_group_stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.50, 0.20, 0.30),
    random_state: int = 42,
    n_restarts: int = 250,
) -> pd.Series:
    """Assign each row to 'train', 'val', or 'test' with zero group leakage.

    Optimizes both target split proportions (50/20/30) and per-class balance across
    splits while keeping every `group_id` intact inside a single split.
    """
    train_r, val_r, test_r = ratios
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    split_names = ("train", "val", "test")
    ratio_vec = np.array([train_r, val_r, test_r], dtype=float)

    classes = sorted(df["target_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    n_total = len(df)

    group_ids: list[str] = []
    group_counts: list[np.ndarray] = []
    group_sizes: list[int] = []
    for gid, grp in df.groupby("group_id", sort=True):
        vec = np.zeros(n_classes, dtype=float)
        for lbl, cnt in grp["target_label"].value_counts().items():
            vec[class_to_idx[lbl]] = float(cnt)
        group_ids.append(str(gid))
        group_counts.append(vec)
        group_sizes.append(len(grp))

    total_class_counts = np.sum(group_counts, axis=0)
    target_split_sizes = ratio_vec * n_total
    target_class_matrix = np.outer(ratio_vec, total_class_counts)

    rng = np.random.default_rng(random_state)
    best_cost = float("inf")
    best_assignment: dict[str, str] = {}

    size_arr = np.array(group_sizes, dtype=float)
    for restart in range(n_restarts):
        if restart == 0:
            order = np.argsort(-size_arr)
        else:
            noise = rng.uniform(0.85, 1.15, size=len(group_ids))
            order = np.argsort(-(size_arr * noise))

        cur_sizes = np.zeros(3, dtype=float)
        cur_classes = np.zeros((3, n_classes), dtype=float)
        assignment: dict[str, str] = {}

        for idx in order:
            g_size = group_sizes[idx]
            g_vec = group_counts[idx]

            best_s = 0
            best_s_score = float("inf")
            for s in range(3):
                new_sizes = cur_sizes.copy()
                new_sizes[s] += g_size
                new_classes = cur_classes.copy()
                new_classes[s] += g_vec

                size_err = np.sum(((new_sizes - target_split_sizes) / np.maximum(target_split_sizes, 1.0)) ** 2)
                class_err = np.sum(
                    ((new_classes - target_class_matrix) / np.maximum(target_class_matrix, 1.0)) ** 2
                )
                score = 2.5 * size_err + class_err
                if score < best_s_score:
                    best_s_score = score
                    best_s = s

            cur_sizes[best_s] += g_size
            cur_classes[best_s] += g_vec
            assignment[group_ids[idx]] = split_names[best_s]

        missing_penalty = 0.0
        for c_idx in range(n_classes):
            if np.sum(cur_classes[:, c_idx] == 0) > 0:
                missing_penalty += 10.0

        final_size_err = np.sum(((cur_sizes - target_split_sizes) / np.maximum(target_split_sizes, 1.0)) ** 2)
        final_class_err = np.sum(
            ((cur_classes - target_class_matrix) / np.maximum(target_class_matrix, 1.0)) ** 2
        )
        total_cost = 2.5 * final_size_err + final_class_err + missing_penalty

        if total_cost < best_cost:
            best_cost = total_cost
            best_assignment = assignment

    return df["group_id"].map(best_assignment)


def verify_split_integrity(df: pd.DataFrame) -> dict[str, Any]:
    """Verify zero group overlap and summarize split/class counts."""
    splits = {}
    group_sets: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        sub = df.loc[df["split"] == name]
        group_sets[name] = set(sub["group_id"].unique())
        splits[name] = {
            "rows": int(len(sub)),
            "pct": round(100.0 * len(sub) / max(len(df), 1), 2),
            "groups": int(len(group_sets[name])),
            "class_counts": {str(k): int(v) for k, v in sub["target_label"].value_counts().items()},
        }

    overlaps = {
        "train_val": len(group_sets["train"] & group_sets["val"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "val_test": len(group_sets["val"] & group_sets["test"]),
    }
    if any(v > 0 for v in overlaps.values()):
        raise RuntimeError(f"Group leakage detected across splits: {overlaps}")

    return {"splits": splits, "group_overlaps": overlaps}


def build_svm_pipeline(config: dict[str, Any], random_state: int = 42) -> Pipeline:
    """Build a scikit-learn TF-IDF + SVM pipeline compatible with Predictor."""
    word_vec = TfidfVectorizer(
        ngram_range=tuple(config["word_ngram_range"]),
        max_features=config["max_word_features"],
        min_df=config["min_df"],
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=config["max_char_features"],
        min_df=config["min_df"],
        sublinear_tf=True,
    )

    model_type = config["model_type"]
    if model_type == "linear_svc_calibrated":
        base_clf = LinearSVC(
            C=config["C"],
            class_weight=config["class_weight"],
            random_state=random_state,
            max_iter=3000,
        )
        clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",
            cv=5,
        )
    elif model_type == "svc_rbf":
        clf = SVC(
            kernel="rbf",
            C=config["C"],
            gamma=config.get("gamma", "scale"),
            class_weight=config["class_weight"],
            probability=True,
            random_state=random_state,
        )
    elif model_type == "svc_linear":
        clf = SVC(
            kernel="linear",
            C=config["C"],
            class_weight=config["class_weight"],
            probability=True,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(
        steps=[
            ("cleaner", TextCleanerTransformer()),
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        ("word_tfidf", word_vec),
                        ("char_tfidf", char_vec),
                    ]
                ),
            ),
            ("clf", clf),
        ]
    )


def get_candidate_configs(include_rbf: bool = True) -> list[dict[str, Any]]:
    """Return candidate SVM hyperparameter configurations for validation tuning."""
    candidates: list[dict[str, Any]] = []
    for c_val in (0.1, 0.3, 0.5, 1.0, 3.0):
        for cw in ("balanced", None):
            candidates.append(
                {
                    "name": f"LinearSVC_C{c_val}_cw-{cw}_ng12",
                    "model_type": "linear_svc_calibrated",
                    "C": c_val,
                    "class_weight": cw,
                    "word_ngram_range": [1, 2],
                    "max_word_features": 12000,
                    "max_char_features": 15000,
                    "min_df": 2,
                }
            )
    for c_val in (0.3, 0.5, 1.0):
        for cw in ("balanced", None):
            candidates.append(
                {
                    "name": f"LinearSVC_C{c_val}_cw-{cw}_ng13",
                    "model_type": "linear_svc_calibrated",
                    "C": c_val,
                    "class_weight": cw,
                    "word_ngram_range": [1, 3],
                    "max_word_features": 15000,
                    "max_char_features": 18000,
                    "min_df": 2,
                }
            )
    if include_rbf:
        for c_val in (1.0, 3.0):
            for cw in ("balanced", None):
                candidates.append(
                    {
                        "name": f"SVC_RBF_C{c_val}_cw-{cw}_ng12",
                        "model_type": "svc_rbf",
                        "C": c_val,
                        "gamma": "scale",
                        "class_weight": cw,
                        "word_ngram_range": [1, 2],
                        "max_word_features": 10000,
                        "max_char_features": 12000,
                        "min_df": 2,
                    }
                )
    return candidates


def save_confusion_matrix_plot(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Save confusion matrix heatmap PNG."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label (SVM)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate group-aware SVM sentiment/emotion classifier."
    )
    parser.add_argument("--input", type=Path, default=None, help="Input CSV path")
    parser.add_argument(
        "--scheme",
        choices=["sentiment3", "ekman7", "main5", "9class"],
        default="sentiment3",
        help="Classification scheme (default: sentiment3)",
    )
    parser.add_argument("--split-file", type=Path, default=None, help="Path to frozen split CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--reuse-split",
        action="store_true",
        help="Load frozen train/val/test assignment from --split-file instead of generating a new one.",
    )
    parser.add_argument(
        "--prepare-split-only",
        action="store_true",
        help="Validate dataset, create and save frozen 50/20/30 group split, then exit before training.",
    )
    parser.add_argument(
        "--skip-rbf",
        action="store_true",
        help="Skip slower RBF kernel SVC candidates during validation tuning.",
    )
    parser.add_argument(
        "--no-refit-train-val",
        action="store_true",
        help="Keep the final model fitted on train-only instead of refitting best config on train+val.",
    )
    args = parser.parse_args()

    # Determine default paths based on chosen scheme
    if args.input is None:
        args.input = DEFAULT_SENTIMENT_INPUT if args.scheme == "sentiment3" else DEFAULT_EMOTION_INPUT
    if args.split_file is None:
        args.split_file = (
            DEFAULT_SENTIMENT_SPLIT if args.scheme == "sentiment3" else DEFAULT_EMOTION_SPLIT
        )

    print(f"[1/5] Loading validated dataset from {args.input} (scheme={args.scheme})...")
    df, data_stats = load_labeled_dataset(args.input, scheme=args.scheme, deduplicate=True)
    print(f"      Data stats: {json.dumps(data_stats, ensure_ascii=False)}")

    print("[2/5] Preparing group-aware 50/20/30 split...")
    if args.reuse_split and args.split_file.is_file():
        split_df = pd.read_csv(args.split_file, dtype={"tweet_id": "string"})
        split_map = dict(zip(split_df["tweet_id"], split_df["split"]))
        df["split"] = df["tweet_id"].map(split_map)
        if df["split"].isna().any():
            missing_count = int(df["split"].isna().sum())
            raise ValueError(
                f"Frozen split file {args.split_file} has {missing_count} unmapped tweet_ids."
            )
        print(f"      Reused frozen split from {args.split_file}")
    else:
        df["split"] = create_group_stratified_split(
            df, ratios=(0.50, 0.20, 0.30), random_state=args.random_state
        )
        args.split_file.parent.mkdir(parents=True, exist_ok=True)
        df[["tweet_id", "group_id", "target_label", "split"]].to_csv(
            args.split_file, index=False, encoding="utf-8-sig"
        )
        print(f"      Saved frozen split to {args.split_file}")

    split_summary = verify_split_integrity(df)
    print(f"      Split summary: {json.dumps(split_summary, ensure_ascii=False)}")

    if args.prepare_split_only:
        print("Exiting after split preparation (--prepare-split-only).")
        return

    train_df = df.loc[df["split"] == "train"]
    val_df = df.loc[df["split"] == "val"]
    test_df = df.loc[df["split"] == "test"]

    X_train, y_train = train_df["raw_text"].tolist(), train_df["target_label"].tolist()
    X_val, y_val = val_df["raw_text"].tolist(), val_df["target_label"].tolist()
    X_test, y_test = test_df["raw_text"].tolist(), test_df["target_label"].tolist()

    print("[3/5] Tuning SVM candidates on validation set (selection metric: Macro-F1)...")
    candidates = get_candidate_configs(include_rbf=not args.skip_rbf)
    tuning_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_config: dict[str, Any] | None = None
    best_train_only_pipeline: Pipeline | None = None

    for cfg in candidates:
        pipe = build_svm_pipeline(cfg, random_state=args.random_state)
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        val_weighted_f1 = float(f1_score(y_val, val_pred, average="weighted", zero_division=0))
        val_acc = float(accuracy_score(y_val, val_pred))

        row = {
            "name": cfg["name"],
            "model_type": cfg["model_type"],
            "C": cfg["C"],
            "class_weight": str(cfg["class_weight"]),
            "word_ngram_range": str(cfg["word_ngram_range"]),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_weighted_f1": round(val_weighted_f1, 4),
            "val_accuracy": round(val_acc, 4),
        }
        tuning_rows.append(row)
        print(
            f"      {cfg['name']:36s} | val_macro_f1={val_macro_f1:.4f} "
            f"| val_weighted_f1={val_weighted_f1:.4f} | val_acc={val_acc:.4f}"
        )

        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_config = cfg
            best_train_only_pipeline = pipe

    assert best_config is not None and best_train_only_pipeline is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuning_df = pd.DataFrame(tuning_rows).sort_values("val_macro_f1", ascending=False)
    tuning_csv = args.output_dir / f"svm_{args.scheme}_tuning_results.csv"
    tuning_df.to_csv(tuning_csv, index=False, encoding="utf-8-sig")
    print(f"      Best config: {best_config['name']} (val_macro_f1={best_score:.4f})")

    print("[4/5] Fitting final model and evaluating ONCE on held-out test set...")
    if args.no_refit_train_val:
        final_pipeline = best_train_only_pipeline
        fit_scope = "train_only"
    else:
        final_pipeline = build_svm_pipeline(best_config, random_state=args.random_state)
        final_pipeline.fit(X_train + X_val, y_train + y_val)
        fit_scope = "train_plus_val"

    test_pred = final_pipeline.predict(X_test)
    labels_ordered = [c for c in SCHEME_CLASSES[args.scheme] if c in set(y_test) | set(test_pred)]
    test_macro_f1 = float(f1_score(y_test, test_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_test, test_pred, average="weighted", zero_division=0))
    test_acc = float(accuracy_score(y_test, test_pred))
    cls_report_dict = classification_report(
        y_test, test_pred, labels=labels_ordered, output_dict=True, zero_division=0
    )
    cls_report_text = classification_report(
        y_test, test_pred, labels=labels_ordered, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_test, test_pred, labels=labels_ordered)

    print("\n=== Held-Out Test Set Evaluation ===")
    print(f"Scheme         : {args.scheme}")
    print(f"Fit scope      : {fit_scope}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test Macro-F1  : {test_macro_f1:.4f}")
    print(f"Test Weighted  : {test_weighted_f1:.4f}\n")
    print(cls_report_text)

    print("[5/5] Saving model pipeline, metadata, and evaluation artifacts...")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    scheme_model_path = args.models_dir / SCHEME_FILENAMES[args.scheme]
    best_model_path = args.models_dir / SCHEME_FILENAMES["best"]
    joblib.dump(final_pipeline, scheme_model_path)
    joblib.dump(final_pipeline, best_model_path)

    cm_df = pd.DataFrame(cm, index=labels_ordered, columns=labels_ordered)
    cm_csv = args.output_dir / f"svm_{args.scheme}_confusion_matrix.csv"
    cm_png = args.output_dir / f"svm_{args.scheme}_confusion_matrix.png"
    cm_df.to_csv(cm_csv, encoding="utf-8-sig")
    save_confusion_matrix_plot(
        cm,
        labels_ordered,
        cm_png,
        title=f"SVM Confusion Matrix ({args.scheme} | Test Macro-F1={test_macro_f1:.3f})",
    )

    report_payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "scheme": args.scheme,
        "fit_scope": fit_scope,
        "random_state": args.random_state,
        "data_stats": data_stats,
        "split_summary": split_summary,
        "best_config": best_config,
        "validation_best_macro_f1": round(best_score, 4),
        "test_metrics": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "per_class": cls_report_dict,
        },
        "ui_styles": {k: DEFAULT_UI_STYLE[k] for k in SCHEME_CLASSES[args.scheme] if k in DEFAULT_UI_STYLE},
    }

    report_json = args.output_dir / f"svm_{args.scheme}_evaluation_report.json"
    report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_json = args.models_dir / "model_metadata.json"
    meta_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model      : {scheme_model_path}")
    print(f"Saved best model : {best_model_path}")
    print(f"Saved metadata   : {meta_json}")
    print(f"Saved report     : {report_json}")
    print(f"Saved CM plot    : {cm_png}")


if __name__ == "__main__":
    main()
