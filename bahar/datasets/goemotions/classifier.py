"""
GoEmotions classifier implementation.

Multilingual emotion classifier using GoEmotions taxonomy.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification

from bahar.datasets.goemotions.result import EmotionResult
from bahar.datasets.goemotions.taxonomy import GOEMOTIONS_EMOTIONS
from bahar.datasets.goemotions.model_adapters import (
    adapt_star_rating_model,
    adapt_binary_sentiment_model,
    load_tokenizer_robust,
)


class GoEmotionsClassifier:
    """
    Multilingual emotion classifier using GoEmotions taxonomy.

    Supports Dutch, Persian, English and other languages via multilingual models.
    """

    def __init__(
        self, model_name: str = "monologg/bert-base-cased-goemotions-original"
    ) -> None:
        """
        Initialize the emotion classifier.

        Args:
            model_name: HuggingFace model name. Default is fine-tuned on GoEmotions.
                       For multilingual support, consider:
                       - "bert-base-multilingual-cased" (requires fine-tuning)
                       - "xlm-roberta-base" (requires fine-tuning)
        """
        self.model_name: str = model_name
        self._model = None
        self._tokenizer = None
        self._label_map: dict[int, str] = {}

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        # Load tokenizer with robust fallback strategies
        self._tokenizer = load_tokenizer_robust(self.model_name)

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        # Create label mapping
        if hasattr(self._model.config, "id2label"):
            self._label_map = self._model.config.id2label
        else:
            # Fallback to GoEmotions taxonomy
            self._label_map = {i: label for i, label in enumerate(GOEMOTIONS_EMOTIONS)}

    def predict(self, text: str, top_k: int = 3) -> EmotionResult:
        """
        Predict emotions for a given text.

        Args:
            text: Input text in any supported language
            top_k: Number of top emotions to return

        Returns:
            EmotionResult with predicted emotions and scores
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        # Type assertions for linter
        assert self._model is not None
        assert self._tokenizer is not None

        # Check if this is a star rating model (5 labels)
        if hasattr(self._model.config, "num_labels") and self._model.config.num_labels == 5:
            # Check if labels are star ratings
            if hasattr(self._model.config, "id2label"):
                first_label = str(self._model.config.id2label.get(0, ""))
                if "star" in first_label.lower() or first_label.isdigit():
                    return adapt_star_rating_model(self.model_name, text, top_k)

        # Check if this is a binary sentiment model (2 labels)
        if hasattr(self._model.config, "num_labels") and self._model.config.num_labels == 2:
            return adapt_binary_sentiment_model(self.model_name, text, top_k)

        # Check if this is a ternary sentiment model (3 labels: recommended/not_recommended/no_idea)
        if hasattr(self._model.config, "num_labels") and self._model.config.num_labels == 3:
            if hasattr(self._model.config, "id2label"):
                labels = [str(v).lower() for v in self._model.config.id2label.values()]
                if "recommended" in labels or "not_recommended" in labels:
                    from bahar.datasets.goemotions.model_adapters import adapt_ternary_sentiment_model
                    return adapt_ternary_sentiment_model(self.model_name, text, top_k)

        # Tokenize input
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # Map predictions to emotion labels
        emotions: dict[str, float] = {}
        for idx, prob in enumerate(probs[0].tolist()):
            label = self._label_map.get(idx, f"label_{idx}")
            emotions[label] = prob

        return EmotionResult(text=text, emotions=emotions, top_k=top_k)

    def predict_batch(
        self, texts: Sequence[str], top_k: int = 3
    ) -> list[EmotionResult]:
        """
        Predict emotions for multiple texts.

        Args:
            texts: List of input texts
            top_k: Number of top emotions to return

        Returns:
            List of EmotionResult objects
        """
        return [self.predict(text, top_k=top_k) for text in texts]

