"""
GoEmotions dataset integration.

Google Research's fine-grained emotion classification dataset.
Reference: https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/
"""

from __future__ import annotations

from bahar.datasets.goemotions.classifier import GoEmotionsClassifier
from bahar.datasets.goemotions.taxonomy import (
    EMOTION_GROUPS,
    GOEMOTIONS_EMOTIONS,
)
from bahar.datasets.goemotions.samples import SAMPLE_TEXTS

__all__ = [
    "GoEmotionsClassifier",
    "GOEMOTIONS_EMOTIONS",
    "EMOTION_GROUPS",
    "SAMPLE_TEXTS",
]

