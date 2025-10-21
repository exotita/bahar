"""
Universal adapter for model predictions.

Provides a unified interface for making predictions with any model.
"""

from __future__ import annotations

from typing import Any

import torch

from bahar.models.metadata import ModelMetadata
from bahar.models.result import UniversalResult


class UniversalAdapter:
    """
    Universal adapter for making predictions with any model.

    Provides a consistent interface regardless of the underlying model type.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        metadata: ModelMetadata
    ) -> None:
        """
        Initialize the universal adapter.

        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            metadata: Model metadata
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move model to device
        if hasattr(self.model, "to"):
            self.model.to(self._device)

    def predict(self, text: str, top_k: int = 3) -> UniversalResult:
        """
        Make a prediction on a single text.

        Args:
            text: Input text
            top_k: Number of top predictions to return

        Returns:
            UniversalResult with predictions
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move inputs to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits
        logits = outputs.logits[0]

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Create predictions dictionary
        predictions = {}
        for idx, prob in enumerate(probabilities):
            label = self.metadata.label_map.get(idx, f"LABEL_{idx}")
            predictions[label] = prob.item()

        # Get top K predictions
        top_indices = torch.topk(probabilities, min(top_k, len(probabilities))).indices
        top_predictions = [
            (self.metadata.label_map.get(idx.item(), f"LABEL_{idx.item()}"),
             probabilities[idx].item())
            for idx in top_indices
        ]

        # Create result
        result = UniversalResult(
            text=text,
            model_id=self.metadata.model_id,
            task_type=self.metadata.task_type,
            predictions=predictions,
            top_predictions=top_predictions,
            metadata={
                "model_name": self.metadata.name,
                "taxonomy": self.metadata.taxonomy,
                "num_labels": self.metadata.num_labels,
            },
            raw_output=outputs,
        )

        return result

    def predict_batch(
        self,
        texts: list[str],
        top_k: int = 3
    ) -> list[UniversalResult]:
        """
        Make predictions on multiple texts.

        Args:
            texts: List of input texts
            top_k: Number of top predictions to return

        Returns:
            List of UniversalResult objects
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move inputs to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits
        logits = outputs.logits

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Create results for each text
        results = []
        for i, text in enumerate(texts):
            # Create predictions dictionary
            predictions = {}
            for idx, prob in enumerate(probabilities[i]):
                label = self.metadata.label_map.get(idx, f"LABEL_{idx}")
                predictions[label] = prob.item()

            # Get top K predictions
            top_indices = torch.topk(
                probabilities[i],
                min(top_k, len(probabilities[i]))
            ).indices
            top_predictions = [
                (self.metadata.label_map.get(idx.item(), f"LABEL_{idx.item()}"),
                 probabilities[i][idx].item())
                for idx in top_indices
            ]

            # Create result
            result = UniversalResult(
                text=text,
                model_id=self.metadata.model_id,
                task_type=self.metadata.task_type,
                predictions=predictions,
                top_predictions=top_predictions,
                metadata={
                    "model_name": self.metadata.name,
                    "taxonomy": self.metadata.taxonomy,
                    "num_labels": self.metadata.num_labels,
                },
                raw_output=None,  # Don't store raw output for batch
            )

            results.append(result)

        return results

    def get_device(self) -> str:
        """Get the device the model is running on."""
        return self._device

    def __repr__(self) -> str:
        return (
            f"UniversalAdapter(model='{self.metadata.model_id}', "
            f"device='{self._device}')"
        )

