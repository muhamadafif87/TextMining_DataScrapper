"""Dashboard-ready inference API for the Perpres 111/2025 SVM emotion models."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from svm.preprocessing import preprocess_text

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

SCHEME_FILENAMES = {
    "ekman7": "svm_ekman7_pipeline.joblib",
    "main5": "svm_5class_pipeline.joblib",
    "9class": "svm_9class_pipeline.joblib",
    "best": "best_svm_pipeline.joblib",
}

DEFAULT_UI_STYLE: dict[str, dict[str, str]] = {
    "marah": {"emoji": "😡", "color_hex": "#D62728", "label_id": "Marah"},
    "senang": {"emoji": "😊", "color_hex": "#2CA02C", "label_id": "Senang"},
    "sedih": {"emoji": "😢", "color_hex": "#1F77B4", "label_id": "Sedih"},
    "takut": {"emoji": "😨", "color_hex": "#FF7F0E", "label_id": "Takut"},
    "jijik": {"emoji": "🤢", "color_hex": "#8C564B", "label_id": "Jijik"},
    "terkejut": {"emoji": "😲", "color_hex": "#9467BD", "label_id": "Terkejut"},
    "netral": {"emoji": "😐", "color_hex": "#7F7F7F", "label_id": "Netral"},
    "sarkas": {"emoji": "😏", "color_hex": "#E377C2", "label_id": "Sarkas"},
    "kecewa": {"emoji": "😞", "color_hex": "#17BECF", "label_id": "Kecewa"},
}


class EmotionPredictor:
    """Wrapper around a trained SVM emotion classification pipeline."""

    def __init__(self, scheme: str = "ekman7", models_dir: str | Path | None = None) -> None:
        if scheme not in SCHEME_FILENAMES:
            raise ValueError(f"Unsupported scheme '{scheme}'. Choose from {list(SCHEME_FILENAMES)}")
        self.scheme = scheme
        self.models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
        model_path = self.models_dir / SCHEME_FILENAMES[scheme]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Run `python -m svm.train_svm` first."
            )
        self.pipeline = joblib.load(model_path)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        meta_path = self.models_dir / "model_metadata.json"
        if meta_path.is_file():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {}

    def _get_style(self, emotion: str) -> dict[str, str]:
        meta_styles = self.metadata.get("emotion_styles", {})
        if emotion in meta_styles:
            return meta_styles[emotion]
        return DEFAULT_UI_STYLE.get(emotion, {"emoji": "💬", "color_hex": "#4A5568", "label_id": emotion.title()})

    def _matched_vocab_features(self, cleaned_text: str) -> list[str]:
        if not cleaned_text:
            return []
        try:
            union = self.pipeline.named_steps["features"]
            word_vec = dict(union.transformer_list)["word_tfidf"]
            vocab = word_vec.vocabulary_
            tokens = cleaned_text.split()
            matched: list[str] = []
            for tok in tokens:
                if tok in vocab and tok not in matched:
                    matched.append(tok)
            for i in range(len(tokens) - 1):
                bg = f"{tokens[i]} {tokens[i + 1]}"
                if bg in vocab and bg not in matched:
                    matched.append(bg)
            return matched[:15]
        except Exception:
            return []

    def predict_single(self, text: str) -> dict:
        """Predict emotion label and class probabilities for one input text."""
        raw_text = "" if text is None else str(text)
        cleaned = preprocess_text(raw_text)
        probs = self.pipeline.predict_proba([raw_text])[0]
        classes = list(self.pipeline.classes_)
        prob_map = {str(cls): round(float(p), 4) for cls, p in zip(classes, probs)}
        sorted_probs = dict(sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True))
        pred_label = next(iter(sorted_probs))
        confidence = sorted_probs[pred_label]
        style = self._get_style(pred_label)

        return {
            "input_text": raw_text,
            "cleaned_text": cleaned,
            "scheme": self.scheme,
            "predicted_emotion": pred_label,
            "emoji": style.get("emoji", "💬"),
            "color_hex": style.get("color_hex", "#4A5568"),
            "confidence": confidence,
            "probabilities": sorted_probs,
            "matched_features": self._matched_vocab_features(cleaned),
        }

    def predict_batch(self, texts: Iterable[str]) -> pd.DataFrame:
        """Predict emotion labels and probabilities for a batch of texts."""
        text_list = ["" if t is None else str(t) for t in texts]
        cleaned_list = [preprocess_text(t) for t in text_list]
        probs = self.pipeline.predict_proba(text_list)
        classes = [str(c) for c in self.pipeline.classes_]

        records: list[dict] = []
        for raw, cln, prob_row in zip(text_list, cleaned_list, probs):
            best_idx = int(prob_row.argmax())
            row: dict[str, object] = {
                "input_text": raw,
                "cleaned_text": cln,
                "predicted_emotion": classes[best_idx],
                "confidence": round(float(prob_row[best_idx]), 4),
            }
            for cls, p in zip(classes, prob_row):
                row[f"prob_{cls}"] = round(float(p), 4)
            records.append(row)
        return pd.DataFrame(records)


def predict_emotion(text: str, scheme: str = "ekman7") -> dict:
    """Convenience one-shot prediction function for dashboard integration."""
    return EmotionPredictor(scheme=scheme).predict_single(text)


def predict_batch(texts: Iterable[str], scheme: str = "ekman7") -> pd.DataFrame:
    """Convenience batch prediction function for dashboard integration."""
    return EmotionPredictor(scheme=scheme).predict_batch(texts)


if __name__ == "__main__":
    predictor = EmotionPredictor(scheme="ekman7")
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        res = predictor.predict_single(query)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        demo_texts = [
            "Ini mah pengalihan isu doang biar rakyat lupa soal korupsi dan ekonomi!",
            "Setuju banget pak Prabowo, kebijakan tepat buat menjaga moral bangsa!",
            "Khawatir banget kalau pasal ini malah memicu persekusi dan konflik horizontal.",
            "Kasihan minoritas yang terus jadi korban perundungan, miris melihatnya.",
            "Najis banget liat kelakuan oknum sampah yang sok suci di sosmed.",
            "Kok tiba-tiba baru heboh sekarang ya padahal Perpres 111 sudah terbit tahun 2025?",
        ]
        print("=== Demo Prediksi Emosi SVM (Skema Ekman 7 Kelas) ===")
        for sample in demo_texts:
            out = predictor.predict_single(sample)
            print(
                f"{out['emoji']} [{out['predicted_emotion']:8s} | {out['confidence']*100:5.1f}%] "
                f"{sample}"
            )
