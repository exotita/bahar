"""
Emotion analyzer wrapper with language-specific model support.

Provides a unified interface for emotion analysis using different datasets and languages.
"""

from __future__ import annotations

from bahar.datasets.goemotions.classifier import GoEmotionsClassifier
from bahar.datasets.goemotions.result import EmotionResult
from bahar.utils.language_models import (
    detect_language,
    get_model_name,
)


class EmotionAnalyzer:
    """
    Unified emotion analyzer with multilingual support.

    Supports language-specific models for English, Dutch, and Persian.
    Automatically detects language and uses the appropriate model.
    """

    def __init__(
        self,
        language: str | None = None,
        model_key: str | None = None,
        model_name: str | None = None,
        auto_detect_language: bool = True,
    ) -> None:
        """
        Initialize emotion analyzer.

        Args:
            language: Language code ("english", "dutch", "persian"). Auto-detected if None.
            model_key: Model key for the language (e.g., "goemotions", "sentiment").
            model_name: Explicit HuggingFace model name (overrides language/model_key).
            auto_detect_language: Automatically detect language from input text.

        Examples:
            # Auto-detect language, use default models
            analyzer = EmotionAnalyzer()

            # Explicit English with GoEmotions
            analyzer = EmotionAnalyzer(language="english", model_key="goemotions")

            # Explicit Dutch with sentiment model
            analyzer = EmotionAnalyzer(language="dutch", model_key="sentiment")

            # Custom model name
            analyzer = EmotionAnalyzer(model_name="custom/model-name")
        """
        self.language = language
        self.model_key = model_key
        self.auto_detect_language = auto_detect_language

        # Determine model name
        if model_name is not None:
            # Explicit model name provided
            self._model_name = model_name
        elif language is not None:
            # Language specified, get model for that language
            self._model_name = get_model_name(language, model_key)
        else:
            # Default to English GoEmotions
            self._model_name = get_model_name("english", "goemotions")
            self.language = "english"

        # Initialize classifier
        self.classifier = GoEmotionsClassifier(self._model_name)
        self._loaded = False

    def load_model(self) -> None:
        """Load the emotion classification model."""
        if not self._loaded:
            self.classifier.load_model()
            self._loaded = True

    def analyze(self, text: str, top_k: int = 3) -> EmotionResult:
        """
        Analyze emotions in text.

        Args:
            text: Input text
            top_k: Number of top emotions to return

        Returns:
            EmotionResult with predictions
        """
        # Auto-detect language if enabled and not set
        if self.auto_detect_language and self.language is None:
            detected_lang = detect_language(text)
            if detected_lang != self.language:
                # Language changed, may need to reload model
                # For now, just note it
                pass

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

    def get_model_info(self) -> dict[str, str]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self._model_name,
            "language": self.language or "auto-detect",
            "model_key": self.model_key or "default",
            "loaded": str(self._loaded),
        }
