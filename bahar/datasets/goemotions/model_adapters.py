"""
Model adapters for different emotion/sentiment models.

Handles different output formats and maps them to a unified EmotionResult format.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from bahar.datasets.goemotions.result import EmotionResult


def load_tokenizer_robust(model_name: str):
    """
    Load tokenizer with multiple fallback strategies.

    Args:
        model_name: HuggingFace model name

    Returns:
        Loaded tokenizer

    Raises:
        RuntimeError: If all loading strategies fail
    """
    # Strategy 1: Try fast tokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except (ValueError, AttributeError, OSError):
        pass

    # Strategy 2: Try slow tokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except (ValueError, AttributeError, OSError):
        pass

    # Strategy 3: Try with trust_remote_code
    try:
        return AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
    except (ValueError, AttributeError, OSError):
        pass

    # Strategy 4: For ALBERT models, try specific tokenizer
    if "albert" in model_name.lower():
        try:
            from transformers import AlbertTokenizer
            return AlbertTokenizer.from_pretrained(model_name)
        except Exception:
            pass

    raise RuntimeError(
        f"Failed to load tokenizer for {model_name}. "
        "This model may have incompatible tokenizer files."
    )


def adapt_star_rating_model(
    model_name: str,
    text: str,
    top_k: int = 3
) -> EmotionResult:
    """
    Adapt star rating models (1-5 stars) to EmotionResult format.

    Maps star ratings to sentiment emotions:
    - 1 star -> disappointment, anger, disgust
    - 2 stars -> disappointment, sadness
    - 3 stars -> neutral
    - 4 stars -> approval, optimism
    - 5 stars -> joy, excitement, gratitude

    Args:
        model_name: HuggingFace model name
        text: Input text
        top_k: Number of top emotions to return

    Returns:
        EmotionResult with mapped emotions
    """
    # Load tokenizer with robust fallback
    tokenizer = load_tokenizer_robust(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Get predictions
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Map stars to emotions
    star_to_emotions = {
        0: {  # 1 star
            "disappointment": 0.4,
            "anger": 0.3,
            "disgust": 0.2,
            "sadness": 0.1,
        },
        1: {  # 2 stars
            "disappointment": 0.5,
            "sadness": 0.3,
            "disapproval": 0.2,
        },
        2: {  # 3 stars
            "neutral": 1.0,
        },
        3: {  # 4 stars
            "approval": 0.5,
            "optimism": 0.3,
            "satisfaction": 0.2,
        },
        4: {  # 5 stars
            "joy": 0.4,
            "excitement": 0.3,
            "gratitude": 0.2,
            "love": 0.1,
        },
    }

    # Calculate weighted emotion scores
    emotion_scores: dict[str, float] = {}
    for star_idx, star_score in enumerate(scores):
        star_score_val = star_score.item()
        if star_score_val > 0.01:  # Only consider significant scores
            emotions = star_to_emotions.get(star_idx, {"neutral": 1.0})
            for emotion, weight in emotions.items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0.0
                emotion_scores[emotion] += star_score_val * weight

    # Normalize scores
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {k: v / total for k, v in emotion_scores.items()}

    # Fill in missing emotions with 0.0
    all_emotions = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]

    for emotion in all_emotions:
        if emotion not in emotion_scores:
            emotion_scores[emotion] = 0.0

    return EmotionResult(text=text, emotions=emotion_scores, top_k=top_k)


def adapt_binary_sentiment_model(
    model_name: str,
    text: str,
    top_k: int = 3
) -> EmotionResult:
    """
    Adapt binary sentiment models (positive/negative) to EmotionResult format.

    Maps binary sentiment to emotions:
    - Positive -> joy, optimism, approval
    - Negative -> sadness, disappointment, anger

    Args:
        model_name: HuggingFace model name
        text: Input text
        top_k: Number of top emotions to return

    Returns:
        EmotionResult with mapped emotions
    """
    # Load tokenizer with robust fallback
    tokenizer = load_tokenizer_robust(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Get predictions
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Map to emotions (assuming [negative, positive] order)
    negative_score = scores[0].item()
    positive_score = scores[1].item() if len(scores) > 1 else 0.0

    emotion_scores = {
        # Negative emotions
        "sadness": negative_score * 0.35,
        "disappointment": negative_score * 0.30,
        "anger": negative_score * 0.20,
        "disapproval": negative_score * 0.15,

        # Positive emotions
        "joy": positive_score * 0.40,
        "optimism": positive_score * 0.30,
        "approval": positive_score * 0.20,
        "gratitude": positive_score * 0.10,

        # Neutral
        "neutral": 0.0,
    }

    # Fill in missing emotions
    all_emotions = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]

    for emotion in all_emotions:
        if emotion not in emotion_scores:
            emotion_scores[emotion] = 0.0

    return EmotionResult(text=text, emotions=emotion_scores, top_k=top_k)


def adapt_ternary_sentiment_model(
    model_name: str,
    text: str,
    top_k: int = 3
) -> EmotionResult:
    """
    Adapt ternary sentiment models (recommended/not_recommended/no_idea) to EmotionResult format.

    Maps ternary sentiment to emotions:
    - Recommended -> joy, approval, optimism
    - Not Recommended -> disappointment, disapproval, sadness
    - No Idea -> neutral, confusion

    Args:
        model_name: HuggingFace model name
        text: Input text
        top_k: Number of top emotions to return

    Returns:
        EmotionResult with mapped emotions
    """
    # Try fast tokenizer first, fallback to slow
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (ValueError, AttributeError):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Get predictions
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # Map to emotions (order: recommended, not_recommended, no_idea)
    # Check model config for actual label order
    label_map = {}
    if hasattr(model.config, "id2label"):
        for idx, label in model.config.id2label.items():
            label_map[idx] = str(label).lower()

    # Find indices for each sentiment
    recommended_idx = None
    not_recommended_idx = None
    no_idea_idx = None

    for idx, label in label_map.items():
        if "recommended" in label and "not" not in label:
            recommended_idx = idx
        elif "not_recommended" in label or "not recommended" in label:
            not_recommended_idx = idx
        elif "no_idea" in label or "no idea" in label or "neutral" in label:
            no_idea_idx = idx

    # Get scores
    recommended_score = scores[recommended_idx].item() if recommended_idx is not None else 0.0
    not_recommended_score = scores[not_recommended_idx].item() if not_recommended_idx is not None else 0.0
    no_idea_score = scores[no_idea_idx].item() if no_idea_idx is not None else 0.0

    emotion_scores = {
        # Positive emotions (from recommended)
        "joy": recommended_score * 0.35,
        "approval": recommended_score * 0.30,
        "optimism": recommended_score * 0.20,
        "gratitude": recommended_score * 0.15,

        # Negative emotions (from not_recommended)
        "disappointment": not_recommended_score * 0.35,
        "disapproval": not_recommended_score * 0.30,
        "sadness": not_recommended_score * 0.20,
        "anger": not_recommended_score * 0.15,

        # Neutral/Ambiguous (from no_idea)
        "neutral": no_idea_score * 0.60,
        "confusion": no_idea_score * 0.40,
    }

    # Fill in missing emotions
    all_emotions = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]

    for emotion in all_emotions:
        if emotion not in emotion_scores:
            emotion_scores[emotion] = 0.0

    return EmotionResult(text=text, emotions=emotion_scores, top_k=top_k)
