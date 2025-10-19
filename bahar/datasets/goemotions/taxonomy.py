"""
GoEmotions taxonomy definitions.

27 emotions + neutral, organized into positive, negative, ambiguous, and neutral groups.
"""

from __future__ import annotations

from typing import Final

# GoEmotions taxonomy: 27 emotions + neutral
GOEMOTIONS_EMOTIONS: Final[list[str]] = [
    # Positive emotions (12)
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
    # Negative emotions (11)
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
    # Ambiguous emotions (4)
    "confusion",
    "curiosity",
    "realization",
    "surprise",
    # Neutral (1)
    "neutral",
]

EMOTION_GROUPS: Final[dict[str, list[str]]] = {
    "positive": [
        "admiration",
        "amusement",
        "approval",
        "caring",
        "desire",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    ],
    "negative": [
        "anger",
        "annoyance",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "fear",
        "grief",
        "nervousness",
        "remorse",
        "sadness",
    ],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}

