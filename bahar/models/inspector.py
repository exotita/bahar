"""
Model inspector for automatic capability detection.

Analyzes models to determine their capabilities and configuration.
"""

from __future__ import annotations

from typing import Any

from bahar.datasets.goemotions.taxonomy import GOEMOTIONS_EMOTIONS
from bahar.models.capabilities import ModelCapabilities


class ModelInspector:
    """
    Inspector for automatically detecting model capabilities.

    Analyzes model configuration and structure to determine what it can do.
    """

    @staticmethod
    def inspect_model(
        model: Any,
        tokenizer: Any,
        config: dict[str, Any]
    ) -> ModelCapabilities:
        """
        Inspect a model and determine its capabilities.

        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            config: Model configuration dictionary

        Returns:
            ModelCapabilities object
        """
        # Detect task type
        task_type = ModelInspector.detect_task_type(config)

        # Extract labels first to get accurate count
        labels = ModelInspector.extract_labels(config)
        num_labels = len(labels) if labels else config.get("num_labels", 0)

        # Detect label type
        label_type = ModelInspector.detect_label_type(config)

        # Get max length
        max_length = ModelInspector.get_max_length(config, tokenizer)

        # Detect architecture
        architecture = config.get("model_type", "unknown")

        # Detect special features
        special_features = ModelInspector.detect_special_features(config)

        # Check multilingual support
        supports_multilingual = ModelInspector.is_multilingual(config, architecture)

        return ModelCapabilities(
            task_type=task_type,
            supports_batch=True,  # Most models support batch
            max_length=max_length,
            num_labels=num_labels,
            label_type=label_type,
            output_format="logits",
            special_features=special_features,
            architecture=architecture,
            requires_preprocessing=False,
            supports_multilingual=supports_multilingual,
        )

    @staticmethod
    def detect_task_type(config: dict[str, Any]) -> str:
        """
        Detect the task type from model configuration.

        Args:
            config: Model configuration

        Returns:
            Task type string
        """
        # Check for explicit task type
        if "task_specific_params" in config and config["task_specific_params"]:
            return list(config["task_specific_params"].keys())[0]

        # Check architecture hints
        architectures = config.get("architectures", [])
        if architectures:
            arch = architectures[0].lower()
            if "forsequenceclassification" in arch:
                return "text-classification"
            if "fortokenclassification" in arch:
                return "token-classification"
            if "forquestionanswering" in arch:
                return "question-answering"
            if "forcausallm" in arch or "formaskedlm" in arch:
                return "text-generation"

        # Default to text classification
        return "text-classification"

    @staticmethod
    def extract_labels(config: dict[str, Any]) -> dict[int, str]:
        """
        Extract label mapping from configuration.

        Args:
            config: Model configuration

        Returns:
            Dictionary mapping label indices to names
        """
        # Try id2label first
        if "id2label" in config and config["id2label"]:
            # Convert string keys to integers
            return {int(k): v for k, v in config["id2label"].items()}

        # Try label2id and reverse it
        if "label2id" in config and config["label2id"]:
            return {v: k for k, v in config["label2id"].items()}

        # Generate default labels based on num_labels
        num_labels = config.get("num_labels", 0)
        if num_labels > 0:
            return {i: f"LABEL_{i}" for i in range(num_labels)}

        # Last resort: empty dict (will be handled by caller)
        return {}

    @staticmethod
    def detect_taxonomy(labels: dict[int, str]) -> str:
        """
        Detect the taxonomy type from labels.

        Args:
            labels: Label mapping

        Returns:
            Taxonomy type string
        """
        label_set = set(labels.values())

        # Check for GoEmotions
        goemotions_set = set(GOEMOTIONS_EMOTIONS)
        if label_set == goemotions_set:
            return "goemotions"

        # Check for common sentiment labels
        sentiment_labels = {"positive", "negative", "neutral"}
        if label_set.issubset(sentiment_labels) or sentiment_labels.issubset(label_set):
            return "sentiment"

        # Check for star ratings
        if all("star" in str(label).lower() for label in label_set):
            return "star_rating"

        # Check for binary classification
        if len(label_set) == 2:
            return "binary"

        return "custom"

    @staticmethod
    def detect_label_type(config: dict[str, Any]) -> str:
        """
        Detect the type of labels (emotion, sentiment, etc.).

        Args:
            config: Model configuration

        Returns:
            Label type string
        """
        labels = ModelInspector.extract_labels(config)
        taxonomy = ModelInspector.detect_taxonomy(labels)

        if taxonomy == "goemotions":
            return "emotion"
        if taxonomy in ["sentiment", "binary", "star_rating"]:
            return "sentiment"

        # Check label names for hints
        label_names = [str(label).lower() for label in labels.values()]

        # Check for emotion keywords
        emotion_keywords = ["joy", "anger", "sadness", "fear", "surprise", "disgust"]
        if any(keyword in " ".join(label_names) for keyword in emotion_keywords):
            return "emotion"

        # Check for sentiment keywords
        sentiment_keywords = ["positive", "negative", "neutral"]
        if any(keyword in " ".join(label_names) for keyword in sentiment_keywords):
            return "sentiment"

        return "custom"

    @staticmethod
    def get_max_length(config: dict[str, Any], tokenizer: Any) -> int:
        """
        Get maximum sequence length.

        Args:
            config: Model configuration
            tokenizer: Loaded tokenizer

        Returns:
            Maximum sequence length
        """
        # Try tokenizer first
        if hasattr(tokenizer, "model_max_length"):
            max_len = tokenizer.model_max_length
            if max_len and max_len < 1000000:  # Sanity check
                return max_len

        # Try config
        for key in ["max_position_embeddings", "n_positions", "max_length"]:
            if key in config:
                return config[key]

        # Default
        return 512

    @staticmethod
    def detect_special_features(config: dict[str, Any]) -> list[str]:
        """
        Detect special features of the model.

        Args:
            config: Model configuration

        Returns:
            List of special feature names
        """
        features = []

        # Check for attention mechanisms
        if config.get("attention_probs_dropout_prob", 0) > 0:
            features.append("attention_dropout")

        # Check for layer normalization
        if "layer_norm_eps" in config:
            features.append("layer_normalization")

        # Check for position embeddings
        if "position_embedding_type" in config:
            features.append(f"position_embedding_{config['position_embedding_type']}")

        return features

    @staticmethod
    def is_multilingual(config: dict[str, Any], architecture: str) -> bool:
        """
        Check if model supports multiple languages.

        Args:
            config: Model configuration
            architecture: Model architecture type

        Returns:
            True if model is multilingual
        """
        multilingual_indicators = [
            "multilingual",
            "xlm",
            "mbert",
            "m-bert",
            "xlm-roberta",
        ]

        model_name = config.get("_name_or_path", "").lower()

        return any(indicator in model_name or indicator in architecture.lower()
                   for indicator in multilingual_indicators)

    @staticmethod
    def get_supported_languages(model_id: str) -> list[str]:
        """
        Attempt to determine supported languages.

        Args:
            model_id: HuggingFace model identifier

        Returns:
            List of language codes (best effort)
        """
        model_id_lower = model_id.lower()

        # Check for language indicators in model ID
        language_map = {
            "english": ["en", "english"],
            "dutch": ["nl", "dutch", "nederlands"],
            "persian": ["fa", "persian", "farsi"],
            "german": ["de", "german", "deutsch"],
            "french": ["fr", "french", "français"],
            "spanish": ["es", "spanish", "español"],
            "chinese": ["zh", "chinese"],
            "arabic": ["ar", "arabic"],
        }

        detected_languages = []
        for lang, indicators in language_map.items():
            if any(indicator in model_id_lower for indicator in indicators):
                detected_languages.append(lang)

        # If multilingual, return common languages
        if any(ind in model_id_lower for ind in ["multilingual", "xlm", "mbert"]):
            return ["english", "dutch", "persian", "german", "french", "spanish"]

        # Default to English if nothing detected
        return detected_languages if detected_languages else ["english"]

