"""
Bahar - Multilingual Emotion and Linguistic Analysis.

A comprehensive text analysis system combining:
- GoEmotions sentiment analysis (28 emotions)
- Linguistic dimensions (formality, tone, intensity, style)
- Multilingual support (English, Dutch, Persian)
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Bahar Team"

# Main exports
from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.analyzers.linguistic_analyzer import LinguisticAnalyzer
from bahar.analyzers.enhanced_analyzer import EnhancedAnalyzer

__all__ = [
    "EmotionAnalyzer",
    "LinguisticAnalyzer",
    "EnhancedAnalyzer",
]

