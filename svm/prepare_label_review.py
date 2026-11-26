"""Create a spreadsheet for human review of rule-based emotion labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from svm.auto_labeler import score_emotions
from svm.preprocessing import is_valid_opinion_tweet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "cleaned_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "cleaned_data_auto_label_review.csv"
EMOTION_CLASSES = (
    "marah",
    "sarkas",
    "kecewa",
    "senang",
    "takut",
    "jijik",
    "sedih",
    "terkejut",
    "netral",
)


def _is_truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_review_frame(input_path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(
        input_path,
        dtype={
            "tweet_id": "string",
            "conversation_id": "string",
            "root_tweet_id": "string",
            "in_reply_to_tweet_id": "string",
        },
        low_memory=False,
    )
    if "raw_text" not in frame.columns:
        raise ValueError(f"Input dataset has no 'raw_text' column: {input_path}")

    suggested_9class: list[str] = []
    suggested_ekman7: list[str] = []
    suggested_main5: list[str] = []
    confidences: list[float | None] = []
    cues: list[str] = []
    validity: list[bool] = []
    filter_reasons: list[str] = []
    scores_by_class: dict[str, list[float | None]] = {
        emotion: [] for emotion in EMOTION_CLASSES
    }

    spam_values = frame.get("is_spam_suspect", pd.Series(False, index=frame.index))
    for text, spam_suspect in zip(frame["raw_text"], spam_values):
        raw_text = "" if pd.isna(text) else str(text)
        is_valid, reason = is_valid_opinion_tweet(raw_text, _is_truthy(spam_suspect))
        validity.append(is_valid)
        filter_reasons.append(reason)

        if not is_valid:
            suggested_9class.append("")
            suggested_ekman7.append("")
            suggested_main5.append("")
            confidences.append(None)
            cues.append("")
            for emotion in EMOTION_CLASSES:
                scores_by_class[emotion].append(None)
            continue

        result = score_emotions(raw_text)
        suggested_9class.append(result.label_9class)
        suggested_ekman7.append(result.label_ekman7)
        suggested_main5.append(result.label_main5)
        confidences.append(result.confidence)
        cues.append(result.matched_cues)
        for emotion in EMOTION_CLASSES:
            scores_by_class[emotion].append(result.scores[emotion])

    frame["is_valid_opinion"] = validity
    frame["filter_reason"] = filter_reasons
    frame["auto_label_9class"] = suggested_9class
    frame["auto_label_ekman7"] = suggested_ekman7
    frame["auto_label_main5"] = suggested_main5
    frame["auto_confidence_heuristic"] = confidences
    frame["auto_matched_cues"] = cues
    for emotion, values in scores_by_class.items():
        frame[f"auto_score_{emotion}"] = values

    frame["manual_label"] = ""
    frame["manual_notes"] = ""
    return frame, sum(validity)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an auto-label review CSV without changing the source dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file (including any manual review edits).",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        parser.error(
            f"Output already exists: {args.output}. Choose another path or pass --force "
            "only if overwriting existing review work is intended."
        )

    review_frame, valid_count = build_review_frame(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    review_frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Input rows: {len(review_frame):,}")
    print(f"Valid opinion rows with suggested labels: {valid_count:,}")
    print(f"Excluded rows: {len(review_frame) - valid_count:,}")
    print(f"Review file: {args.output}")


if __name__ == "__main__":
    main()
