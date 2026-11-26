"""SVM Emotion Classification package for Perpres No. 111 Tahun 2025 public sentiment."""
from __future__ import annotations

from svm.predictor import EmotionPredictor, predict_batch, predict_emotion
from svm.preprocessing import TextCleanerTransformer, is_valid_opinion_tweet, preprocess_text

__all__ = [
    "EmotionPredictor",
    "TextCleanerTransformer",
    "is_valid_opinion_tweet",
    "predict_batch",
    "predict_emotion",
    "preprocess_text",
]
