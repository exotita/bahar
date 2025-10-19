"""
Emotion analyzer wrapper.

Provides a unified interface for emotion analysis using different datasets.
"""

from __future__ import annotations

from bahar.datasets.goemotions.classifier import GoEmotionsClassifier
from bahar.datasets.goemotions.result import EmotionResult


class EmotionAnalyzer:
    """
    Unified emotion analyzer.

    Currently supports GoEmotions dataset.
    Can be extended to support other emotion datasets.
    """

    def __init__(
        self,
        dataset: str = "goemotions",
        model_name: str | None = None,
    ) -> None:
        """
        Initialize emotion analyzer.

        Args:
            dataset: Dataset to use ("goemotions", etc.)
            model_name: Optional custom model name
        """
        self.dataset: str = dataset

        if dataset == "goemotions":
            if model_name is None:
                model_name = "monologg/bert-base-cased-goemotions-original"
            self.classifier = GoEmotionsClassifier(model_name)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    def load_model(self) -> None:
        """Load the emotion classification model."""
        self.classifier.load_model()

    def analyze(self, text: str, top_k: int = 3) -> EmotionResult:
        """
        Analyze emotions in text.

        Args:
            text: Input text
            top_k: Number of top emotions to return

        Returns:
            EmotionResult with predictions
        """
        return self.classifier.predict(text, top_k=top_k)

    def analyze_batch(
        self, texts: list[str], top_k: int = 3
    ) -> list[EmotionResult]:
        """
        Analyze emotions in multiple texts.

        Args:
            texts: List of input texts
            top_k: Number of top emotions to return

        Returns:
            List of EmotionResult objects
        """
        return self.classifier.predict_batch(texts, top_k=top_k)

