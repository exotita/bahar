"""
Universal model loader for HuggingFace models.

Dynamically loads any compatible model from HuggingFace Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForSequenceClassification

from bahar.datasets.goemotions.model_adapters import load_tokenizer_robust
from bahar.models.metadata import ModelMetadata


class UniversalModelLoader:
    """
    Universal loader for HuggingFace models.

    Supports dynamic loading of any sequence classification model.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """
        Initialize the universal model loader.

        Args:
            cache_dir: Directory for caching models. Uses HuggingFace default if None.
        """
        self.cache_dir = cache_dir
        self._loaded_models: dict[str, tuple[Any, Any, Any]] = {}

    def load_model(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        **kwargs: Any
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Load a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model identifier
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model loading

        Returns:
            Tuple of (model, tokenizer, config_dict)

        Raises:
            RuntimeError: If model loading fails
        """
        # Check cache
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        try:
            # Load configuration
            config = AutoConfig.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
            )

            # Load tokenizer with robust fallback
            tokenizer = load_tokenizer_robust(model_id)

            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs
            )

            # Extract config as dict
            config_dict = config.to_dict()

            # Cache loaded model
            self._loaded_models[model_id] = (model, tokenizer, config_dict)

            return model, tokenizer, config_dict

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_id}': {e}"
            ) from e

    def load_from_metadata(
        self,
        metadata: ModelMetadata,
        **kwargs: Any
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Load a model using metadata.

        Args:
            metadata: Model metadata
            **kwargs: Additional arguments for model loading

        Returns:
            Tuple of (model, tokenizer, config_dict)
        """
        # Merge custom config with kwargs
        load_kwargs = {**metadata.custom_config, **kwargs}

        return self.load_model(metadata.model_id, **load_kwargs)

    def validate_model(self, model_id: str) -> tuple[bool, str]:
        """
        Validate that a model can be loaded.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Try to load config only (lightweight check)
            AutoConfig.from_pretrained(model_id, cache_dir=self.cache_dir)
            return True, ""
        except Exception as e:
            return False, str(e)

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """
        Get basic information about a model without loading it.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            Dictionary with model information

        Raises:
            RuntimeError: If model info cannot be retrieved
        """
        try:
            config = AutoConfig.from_pretrained(model_id, cache_dir=self.cache_dir)

            info = {
                "model_id": model_id,
                "model_type": config.model_type,
                "num_labels": getattr(config, "num_labels", 0),
                "max_position_embeddings": getattr(config, "max_position_embeddings", 512),
                "vocab_size": getattr(config, "vocab_size", 0),
                "hidden_size": getattr(config, "hidden_size", 0),
                "num_hidden_layers": getattr(config, "num_hidden_layers", 0),
                "num_attention_heads": getattr(config, "num_attention_heads", 0),
            }

            # Add label mapping if available
            if hasattr(config, "id2label"):
                info["id2label"] = config.id2label
            if hasattr(config, "label2id"):
                info["label2id"] = config.label2id

            return info

        except Exception as e:
            raise RuntimeError(
                f"Failed to get model info for '{model_id}': {e}"
            ) from e

    def unload_model(self, model_id: str) -> None:
        """
        Unload a model from cache.

        Args:
            model_id: HuggingFace model identifier
        """
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]

    def clear_cache(self) -> None:
        """Clear all loaded models from cache."""
        self._loaded_models.clear()

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return list(self._loaded_models.keys())

    def __repr__(self) -> str:
        return f"UniversalModelLoader(loaded={len(self._loaded_models)})"

