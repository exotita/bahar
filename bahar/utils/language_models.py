"""
Language-specific model configuration for emotion analysis.

Provides model mappings and language detection for Dutch, Persian, and English.
"""

from __future__ import annotations

from typing import Final


# Language-specific model configurations
LANGUAGE_MODELS: Final[dict[str, dict[str, str]]] = {
    "english": {
        # GoEmotions - 28 fine-grained emotions
        "goemotions": "monologg/bert-base-cased-goemotions-original",
        # Alternative: SamLowe's RoBERTa model
        "goemotions-roberta": "SamLowe/roberta-base-go_emotions",
        # Alternative: j-hartmann's model
        "goemotions-ekman": "j-hartmann/emotion-english-distilroberta-base",
    },
    "dutch": {
        # BERT multilingual sentiment (1-5 stars)
        "sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
        # BERTje - Dutch BERT
        "bertje-sentiment": "DTAI-KULeuven/robbert-v2-dutch-sentiment",
        # Alternative: wietsedv's Dutch emotion model
        "emotion": "wietsedv/bert-base-dutch-cased-finetuned-sentiment",
    },
    "persian": {
        # ALBERT Persian sentiment
        "sentiment": "m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-binary",
        # ParsBERT sentiment
        "parsbert": "HooshvareLab/bert-fa-base-uncased-sentiment-digikala",
        # Alternative: ParsBERT base
        "parsbert-emotion": "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood",
    },
}


# Default models for each language
DEFAULT_MODELS: Final[dict[str, str]] = {
    "english": "goemotions",
    "dutch": "sentiment",
    "persian": "sentiment",
}


# Language detection patterns (simple keyword-based)
LANGUAGE_PATTERNS: Final[dict[str, list[str]]] = {
    "dutch": [
        "het", "een", "is", "van", "de", "en", "dit", "dat", "zijn", "was",
        "maar", "niet", "hij", "heeft", "zijn", "op", "voor", "met", "als", "er"
    ],
    "persian": [
        "است", "این", "که", "را", "از", "به", "در", "با", "من", "و",
        "یک", "برای", "هم", "کرد", "شد", "دارد", "خود", "تا", "کند", "بود"
    ],
    "english": [
        "the", "is", "a", "an", "and", "of", "to", "in", "for", "that",
        "it", "was", "with", "as", "are", "this", "but", "not", "have", "from"
    ],
}


def detect_language(text: str) -> str:
    """
    Detect the language of input text using pattern matching.

    Args:
        text: Input text

    Returns:
        Detected language: "english", "dutch", "persian", or "english" (default)
    """
    if not text or not text.strip():
        return "english"

    text_lower = text.lower()
    words = text_lower.split()

    if not words:
        return "english"

    # Check for Persian characters (Unicode range)
    if any('\u0600' <= char <= '\u06FF' for char in text):
        return "persian"

    # Count pattern matches for each language
    scores: dict[str, int] = {}
    for lang, patterns in LANGUAGE_PATTERNS.items():
        score = sum(1 for word in words if word in patterns)
        scores[lang] = score

    # Return language with highest score, default to English
    if max(scores.values()) == 0:
        return "english"

    return max(scores.items(), key=lambda x: x[1])[0]


def get_model_name(language: str, model_key: str | None = None) -> str:
    """
    Get the HuggingFace model name for a language and model key.

    Args:
        language: Language code ("english", "dutch", "persian")
        model_key: Model key (e.g., "goemotions", "sentiment"). If None, uses default.

    Returns:
        HuggingFace model name

    Raises:
        ValueError: If language or model_key is invalid
    """
    if language not in LANGUAGE_MODELS:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {list(LANGUAGE_MODELS.keys())}"
        )

    if model_key is None:
        model_key = DEFAULT_MODELS[language]

    if model_key not in LANGUAGE_MODELS[language]:
        raise ValueError(
            f"Unsupported model '{model_key}' for {language}. "
            f"Available: {list(LANGUAGE_MODELS[language].keys())}"
        )

    return LANGUAGE_MODELS[language][model_key]


def get_available_models(language: str) -> dict[str, str]:
    """
    Get all available models for a language.

    Args:
        language: Language code

    Returns:
        Dictionary of model_key -> model_name
    """
    if language not in LANGUAGE_MODELS:
        return {}
    return LANGUAGE_MODELS[language].copy()


def get_supported_languages() -> list[str]:
    """
    Get list of supported languages.

    Returns:
        List of language codes
    """
    return list(LANGUAGE_MODELS.keys())

