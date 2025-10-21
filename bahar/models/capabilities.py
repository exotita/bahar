"""
Model capabilities detection and representation.

Defines what a model can do and how to interact with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelCapabilities:
    """
    Represents the capabilities and configuration of a model.

    Attributes:
        task_type: Type of task the model performs
        supports_batch: Whether the model supports batch processing
        max_length: Maximum input sequence length
        num_labels: Number of output labels
        label_type: Type of labels (emotion, sentiment, custom, etc.)
        output_format: Format of model output
        special_features: Additional capabilities or features
        architecture: Model architecture type
        requires_preprocessing: Whether input requires special preprocessing
        supports_multilingual: Whether model supports multiple languages
    """

    task_type: str
    supports_batch: bool = True
    max_length: int = 512
    num_labels: int = 0
    label_type: str = "custom"
    output_format: str = "logits"
    special_features: list[str] = field(default_factory=list)
    architecture: str | None = None
    requires_preprocessing: bool = False
    supports_multilingual: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert capabilities to dictionary."""
        return {
            "task_type": self.task_type,
            "supports_batch": self.supports_batch,
            "max_length": self.max_length,
            "num_labels": self.num_labels,
            "label_type": self.label_type,
            "output_format": self.output_format,
            "special_features": self.special_features,
            "architecture": self.architecture,
            "requires_preprocessing": self.requires_preprocessing,
            "supports_multilingual": self.supports_multilingual,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCapabilities:
        """Create capabilities from dictionary."""
        return cls(**data)

    def is_compatible_with(self, task: str) -> bool:
        """Check if capabilities are compatible with a task."""
        return self.task_type == task

    def __repr__(self) -> str:
        return (
            f"ModelCapabilities(task_type='{self.task_type}', "
            f"num_labels={self.num_labels}, label_type='{self.label_type}')"
        )


# Common task types
TASK_TYPES: dict[str, str] = {
    "text-classification": "Text Classification",
    "token-classification": "Token Classification (NER)",
    "question-answering": "Question Answering",
    "text-generation": "Text Generation",
    "fill-mask": "Fill Mask",
    "sentiment-analysis": "Sentiment Analysis",
    "zero-shot-classification": "Zero-Shot Classification",
}


# Common label types
LABEL_TYPES: dict[str, str] = {
    "emotion": "Emotion Classification",
    "sentiment": "Sentiment Analysis",
    "topic": "Topic Classification",
    "intent": "Intent Detection",
    "custom": "Custom Classification",
}

