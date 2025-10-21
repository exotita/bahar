"""
Bahar - Multilingual Emotion and Linguistic Analysis.

A comprehensive text analysis system combining:
- GoEmotions sentiment analysis (28 emotions)
- Linguistic dimensions (formality, tone, intensity, style)
- Advanced linguistic analysis (semantics, morphology, embeddings, discourse)
- Multilingual support (English, Dutch, Persian)
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Bahar Team"

# Main exports - Emotion & Linguistic Analysis
from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.analyzers.linguistic_analyzer import LinguisticAnalyzer
from bahar.analyzers.enhanced_analyzer import EnhancedAnalyzer

# Advanced Linguistic Analysis (NEW)
from bahar.analyzers.semantic_analyzer import SemanticAnalyzer
from bahar.analyzers.morphology_analyzer import MorphologyAnalyzer
from bahar.analyzers.embedding_analyzer import EmbeddingAnalyzer
from bahar.analyzers.discourse_analyzer import DiscourseAnalyzer
from bahar.analyzers.advanced_analyzer import AdvancedLinguisticAnalyzer

__all__ = [
    # Core analyzers
    "EmotionAnalyzer",
    "LinguisticAnalyzer",
    "EnhancedAnalyzer",
    # Advanced analyzers
    "SemanticAnalyzer",
    "MorphologyAnalyzer",
    "EmbeddingAnalyzer",
    "DiscourseAnalyzer",
    "AdvancedLinguisticAnalyzer",
]

