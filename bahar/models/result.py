"""
Universal result format for model predictions.

Provides a flexible result structure that can be converted to various formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from bahar.datasets.goemotions.result import EmotionResult


@dataclass
class UniversalResult:
    """
    Universal result format for any model prediction.

    Provides a flexible structure that can represent results from any model type
    and convert to legacy formats for backward compatibility.

    Attributes:
        text: Input text that was analyzed
        model_id: HuggingFace model identifier used
        task_type: Type of task performed
        predictions: Dictionary mapping labels to scores
        top_predictions: List of top N predictions
        metadata: Additional metadata about the prediction
        raw_output: Original model output (for debugging/analysis)
        timestamp: When the prediction was made
    """

    text: str
    model_id: str
    task_type: str
    predictions: dict[str, float]
    top_predictions: list[tuple[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_output: Any | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def get_top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """
        Get top K predictions.

        Args:
            k: Number of top predictions to return

        Returns:
            List of (label, score) tuples
        """
        return self.top_predictions[:k]

    def get_prediction(self, label: str) -> float:
        """
        Get score for a specific label.

        Args:
            label: Label name

        Returns:
            Score for the label, or 0.0 if not found
        """
        return self.predictions.get(label, 0.0)

    def to_emotion_result(self) -> EmotionResult:
        """
        Convert to legacy EmotionResult format.

        Returns:
            EmotionResult object
        """
        top_k = len(self.top_predictions)
        return EmotionResult(
            text=self.text,
            emotions=self.predictions,
            top_k=top_k
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        data = {
            "text": self.text,
            "model_id": self.model_id,
            "task_type": self.task_type,
            "predictions": self.predictions,
            "top_predictions": [
                {"label": label, "score": score}
                for label, score in self.top_predictions
            ],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
        return data

    def to_json(self, indent: int | None = 2) -> str:
        """
        Convert to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniversalResult:
        """
        Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            UniversalResult object
        """
        # Convert top_predictions from dict format to tuple format
        if "top_predictions" in data and isinstance(data["top_predictions"], list):
            if data["top_predictions"] and isinstance(data["top_predictions"][0], dict):
                data["top_predictions"] = [
                    (item["label"], item["score"])
                    for item in data["top_predictions"]
                ]

        # Convert timestamp
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)

    def get_sentiment_category(self) -> str:
        """
        Determine overall sentiment category from predictions.

        Returns:
            Sentiment category: "positive", "negative", "neutral", or "mixed"
        """
        # If predictions contain sentiment labels directly
        sentiment_labels = ["positive", "negative", "neutral"]
        sentiment_scores = {
            label: score
            for label, score in self.predictions.items()
            if label.lower() in sentiment_labels
        }

        if sentiment_scores:
            return max(sentiment_scores.items(), key=lambda x: x[1])[0]

        # Try to infer from emotion labels
        positive_emotions = [
            "joy", "love", "optimism", "gratitude", "admiration", "amusement",
            "approval", "caring", "desire", "excitement", "pride", "relief"
        ]
        negative_emotions = [
            "anger", "annoyance", "disappointment", "disapproval", "disgust",
            "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
        ]

        positive_score = sum(
            score for label, score in self.predictions.items()
            if label.lower() in positive_emotions
        )
        negative_score = sum(
            score for label, score in self.predictions.items()
            if label.lower() in negative_emotions
        )

        if positive_score > negative_score * 1.5:
            return "positive"
        if negative_score > positive_score * 1.5:
            return "negative"
        if abs(positive_score - negative_score) < 0.1:
            return "neutral"

        return "mixed"

    def __repr__(self) -> str:
        top_3 = self.get_top_k(3)
        predictions_str = ", ".join(f"{label}={score:.3f}" for label, score in top_3)
        return (
            f"UniversalResult(model='{self.model_id}', "
            f"top_predictions=[{predictions_str}])"
        )

