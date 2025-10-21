"""
Morphology and Phonology Analyzer.

This module provides comprehensive morphological and phonological analysis:
- Morphological analysis (morphemes, affixes, compounds)
- Phonological features (syllables, phoneme distribution)
- Lemmatization and stemming comparison
- Morphological complexity metrics

Academic research focused with statistical metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import spacy
from spacy.tokens import Doc
import pyphen


@dataclass
class MorphemeAnalysis:
    """Morpheme analysis for a word."""

    word: str
    lemma: str
    morphemes: list[str]  # Estimated morpheme segments
    affixes: list[str]  # Detected affixes
    is_compound: bool
    morphological_features: dict[str, str]  # spaCy morph features

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "word": self.word,
            "lemma": self.lemma,
            "morphemes": self.morphemes,
            "morpheme_count": len(self.morphemes),
            "affixes": self.affixes,
            "is_compound": self.is_compound,
            "morphological_features": self.morphological_features,
        }


@dataclass
class PhonologicalFeatures:
    """Phonological features for text."""

    word: str
    syllables: list[str]
    syllable_count: int
    consonant_count: int
    vowel_count: int
    consonant_clusters: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "word": self.word,
            "syllables": self.syllables,
            "syllable_count": self.syllable_count,
            "consonant_count": self.consonant_count,
            "vowel_count": self.vowel_count,
            "cv_ratio": self.consonant_count / self.vowel_count if self.vowel_count > 0 else 0,
            "consonant_clusters": self.consonant_clusters,
        }


@dataclass
class MorphologyFeatures:
    """Complete morphological and phonological analysis results."""

    text: str
    morpheme_analyses: list[MorphemeAnalysis] = field(default_factory=list)
    phonological_features: list[PhonologicalFeatures] = field(default_factory=list)

    # Statistics
    morphemes_per_word: float = 0.0
    morphological_complexity: float = 0.0
    derivational_ratio: float = 0.0
    inflectional_ratio: float = 0.0
    syllables_per_word: float = 0.0
    consonant_vowel_ratio: float = 0.0
    phonological_complexity: float = 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "text_length": len(self.text),
            "word_count": len(self.text.split()),
            "analyzed_words": len(self.morpheme_analyses),
            "morphemes_per_word": self.morphemes_per_word,
            "morphological_complexity": self.morphological_complexity,
            "syllables_per_word": self.syllables_per_word,
            "consonant_vowel_ratio": self.consonant_vowel_ratio,
            "phonological_complexity": self.phonological_complexity,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "text": self.text,
            "morpheme_analyses": [ma.to_dict() for ma in self.morpheme_analyses],
            "phonological_features": [pf.to_dict() for pf in self.phonological_features],
            "statistics": {
                "morphemes_per_word": self.morphemes_per_word,
                "morphological_complexity": self.morphological_complexity,
                "derivational_ratio": self.derivational_ratio,
                "inflectional_ratio": self.inflectional_ratio,
                "syllables_per_word": self.syllables_per_word,
                "consonant_vowel_ratio": self.consonant_vowel_ratio,
                "phonological_complexity": self.phonological_complexity,
            }
        }


class MorphologyAnalyzer:
    """
    Analyzer for morphology and phonology.

    Provides:
    - Morphological analysis (morphemes, affixes, compounds)
    - Phonological features (syllables, phoneme distribution)
    - Morphological complexity metrics
    - Phonological complexity metrics

    Academic research focused with comprehensive statistics.
    """

    # Common English affixes for detection
    PREFIXES = {
        'un', 're', 'in', 'im', 'dis', 'en', 'non', 'pre', 'pro', 'anti',
        'de', 'over', 'mis', 'sub', 'inter', 'fore', 'under', 'counter'
    }

    SUFFIXES = {
        'ed', 'ing', 'ly', 'er', 'est', 'tion', 'sion', 'ness', 'ment',
        'ful', 'less', 'able', 'ible', 'ous', 'ious', 'al', 'ial', 'ic',
        'ive', 'ize', 'ise', 'ify', 'en', 'ate'
    }

    VOWELS = set('aeiouAEIOU')

    def __init__(self, language: str = "english") -> None:
        """
        Initialize morphology analyzer.

        Args:
            language: Language code (english, dutch, persian)
        """
        self.language = language
        self._nlp: Any = None
        self._pyphen_dic: Any = None
        self._loaded = False

    def load_model(self) -> None:
        """Load required models."""
        if self._loaded:
            return

        # Load spaCy model with morphology
        model_map = {
            "english": "en_core_web_lg",
            "dutch": "nl_core_news_lg",
            "persian": "en_core_web_lg",  # Fallback
        }
        model_name = model_map.get(self.language, "en_core_web_lg")

        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            # Fallback to small model
            model_name = model_name.replace("_lg", "_sm")
            self._nlp = spacy.load(model_name)

        # Load pyphen dictionary for syllabification
        lang_map = {
            "english": "en_US",
            "dutch": "nl_NL",
            "persian": "en_US",  # Fallback
        }
        pyphen_lang = lang_map.get(self.language, "en_US")
        self._pyphen_dic = pyphen.Pyphen(lang=pyphen_lang)

        self._loaded = True

    def analyze(self, text: str) -> MorphologyFeatures:
        """
        Perform comprehensive morphological and phonological analysis.

        Args:
            text: Input text to analyze

        Returns:
            MorphologyFeatures with complete analysis
        """
        if not self._loaded:
            self.load_model()

        # Process with spaCy
        doc = self._nlp(text)

        # Morphological analysis
        morpheme_analyses = self._analyze_morphology(doc)

        # Phonological analysis
        phonological_features = self._analyze_phonology(doc)

        # Compute statistics
        morphemes_per_word = self._compute_morphemes_per_word(morpheme_analyses)
        morphological_complexity = self._compute_morphological_complexity(morpheme_analyses)
        derivational_ratio = self._compute_derivational_ratio(morpheme_analyses)
        inflectional_ratio = self._compute_inflectional_ratio(morpheme_analyses)
        syllables_per_word = self._compute_syllables_per_word(phonological_features)
        cv_ratio = self._compute_cv_ratio(phonological_features)
        phon_complexity = self._compute_phonological_complexity(phonological_features)

        return MorphologyFeatures(
            text=text,
            morpheme_analyses=morpheme_analyses,
            phonological_features=phonological_features,
            morphemes_per_word=morphemes_per_word,
            morphological_complexity=morphological_complexity,
            derivational_ratio=derivational_ratio,
            inflectional_ratio=inflectional_ratio,
            syllables_per_word=syllables_per_word,
            consonant_vowel_ratio=cv_ratio,
            phonological_complexity=phon_complexity,
        )

    def _analyze_morphology(self, doc: Doc) -> list[MorphemeAnalysis]:
        """Analyze morphology of words."""
        analyses = []

        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue

            word = token.text.lower()
            lemma = token.lemma_.lower()

            # Extract morphological features from spaCy
            morph_features = {}
            if token.morph:
                for feature in token.morph:
                    key, value = feature.split('=') if '=' in feature else (feature, '')
                    morph_features[key] = value

            # Detect affixes
            affixes = self._detect_affixes(word)

            # Estimate morphemes (simplified)
            morphemes = self._estimate_morphemes(word, lemma, affixes)

            # Check if compound
            is_compound = self._is_compound(word)

            analysis = MorphemeAnalysis(
                word=word,
                lemma=lemma,
                morphemes=morphemes,
                affixes=affixes,
                is_compound=is_compound,
                morphological_features=morph_features,
            )
            analyses.append(analysis)

        return analyses

    def _analyze_phonology(self, doc: Doc) -> list[PhonologicalFeatures]:
        """Analyze phonological features."""
        features_list = []

        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue

            word = token.text.lower()

            # Syllabification
            syllables = self._syllabify(word)

            # Count consonants and vowels
            consonants = sum(1 for c in word if c.isalpha() and c not in self.VOWELS)
            vowels = sum(1 for c in word if c in self.VOWELS)

            # Detect consonant clusters
            clusters = self._detect_consonant_clusters(word)

            features = PhonologicalFeatures(
                word=word,
                syllables=syllables,
                syllable_count=len(syllables),
                consonant_count=consonants,
                vowel_count=vowels,
                consonant_clusters=clusters,
            )
            features_list.append(features)

        return features_list

    def _detect_affixes(self, word: str) -> list[str]:
        """Detect prefixes and suffixes."""
        affixes = []

        # Check prefixes
        for prefix in self.PREFIXES:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                affixes.append(f"prefix:{prefix}")

        # Check suffixes
        for suffix in self.SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                affixes.append(f"suffix:{suffix}")

        return affixes

    def _estimate_morphemes(
        self,
        word: str,
        lemma: str,
        affixes: list[str]
    ) -> list[str]:
        """Estimate morpheme segmentation (simplified)."""
        morphemes = []

        # Start with lemma as root
        remaining = word

        # Remove detected prefixes
        for affix in affixes:
            if affix.startswith('prefix:'):
                prefix = affix.split(':')[1]
                if remaining.startswith(prefix):
                    morphemes.append(prefix)
                    remaining = remaining[len(prefix):]

        # Add root (what's left after prefixes, before suffixes)
        root = remaining
        for affix in affixes:
            if affix.startswith('suffix:'):
                suffix = affix.split(':')[1]
                if root.endswith(suffix):
                    root = root[:-len(suffix)]

        if root:
            morphemes.append(root)

        # Add suffixes
        for affix in affixes:
            if affix.startswith('suffix:'):
                suffix = affix.split(':')[1]
                morphemes.append(suffix)

        # If no morphemes detected, treat whole word as single morpheme
        if not morphemes:
            morphemes = [word]

        return morphemes

    def _is_compound(self, word: str) -> bool:
        """Check if word is likely a compound (simplified heuristic)."""
        # Very simple heuristic: long words with multiple capital letters
        # or words with hyphens
        if '-' in word:
            return True
        if len(word) > 10 and sum(1 for c in word if c.isupper()) > 1:
            return True
        return False

    def _syllabify(self, word: str) -> list[str]:
        """Syllabify word using pyphen."""
        if not self._pyphen_dic:
            return [word]

        syllabified = self._pyphen_dic.inserted(word, hyphen='·')
        return syllabified.split('·')

    def _detect_consonant_clusters(self, word: str) -> list[str]:
        """Detect consonant clusters."""
        clusters = []
        current_cluster = []

        for char in word:
            if char.isalpha() and char not in self.VOWELS:
                current_cluster.append(char)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(''.join(current_cluster))
                current_cluster = []

        # Check final cluster
        if len(current_cluster) >= 2:
            clusters.append(''.join(current_cluster))

        return clusters

    def _compute_morphemes_per_word(
        self,
        analyses: list[MorphemeAnalysis]
    ) -> float:
        """Compute average morphemes per word."""
        if not analyses:
            return 0.0
        total_morphemes = sum(len(a.morphemes) for a in analyses)
        return total_morphemes / len(analyses)

    def _compute_morphological_complexity(
        self,
        analyses: list[MorphemeAnalysis]
    ) -> float:
        """
        Compute morphological complexity index.

        Based on:
        - Number of morphemes
        - Number of affixes
        - Presence of compounds
        """
        if not analyses:
            return 0.0

        complexity_scores = []
        for analysis in analyses:
            score = 0.0

            # More morphemes = more complex
            score += (len(analysis.morphemes) - 1) * 0.3

            # Affixes add complexity
            score += len(analysis.affixes) * 0.2

            # Compounds are complex
            if analysis.is_compound:
                score += 0.5

            complexity_scores.append(min(score, 1.0))  # Cap at 1.0

        return sum(complexity_scores) / len(complexity_scores)

    def _compute_derivational_ratio(
        self,
        analyses: list[MorphemeAnalysis]
    ) -> float:
        """Compute ratio of derivational affixes."""
        if not analyses:
            return 0.0

        # Simplified: count prefixes and certain suffixes as derivational
        derivational_suffixes = {'tion', 'sion', 'ness', 'ment', 'ful', 'less', 'ize', 'ify'}

        total_affixes = 0
        derivational_count = 0

        for analysis in analyses:
            for affix in analysis.affixes:
                total_affixes += 1
                if affix.startswith('prefix:'):
                    derivational_count += 1
                elif affix.startswith('suffix:'):
                    suffix = affix.split(':')[1]
                    if suffix in derivational_suffixes:
                        derivational_count += 1

        return derivational_count / total_affixes if total_affixes > 0 else 0.0

    def _compute_inflectional_ratio(
        self,
        analyses: list[MorphemeAnalysis]
    ) -> float:
        """Compute ratio of inflectional affixes."""
        if not analyses:
            return 0.0

        # Simplified: count certain suffixes as inflectional
        inflectional_suffixes = {'ed', 'ing', 'er', 'est', 's'}

        total_affixes = 0
        inflectional_count = 0

        for analysis in analyses:
            for affix in analysis.affixes:
                total_affixes += 1
                if affix.startswith('suffix:'):
                    suffix = affix.split(':')[1]
                    if suffix in inflectional_suffixes:
                        inflectional_count += 1

        return inflectional_count / total_affixes if total_affixes > 0 else 0.0

    def _compute_syllables_per_word(
        self,
        features_list: list[PhonologicalFeatures]
    ) -> float:
        """Compute average syllables per word."""
        if not features_list:
            return 0.0
        total_syllables = sum(f.syllable_count for f in features_list)
        return total_syllables / len(features_list)

    def _compute_cv_ratio(
        self,
        features_list: list[PhonologicalFeatures]
    ) -> float:
        """Compute consonant-vowel ratio."""
        if not features_list:
            return 0.0

        total_consonants = sum(f.consonant_count for f in features_list)
        total_vowels = sum(f.vowel_count for f in features_list)

        return total_consonants / total_vowels if total_vowels > 0 else 0.0

    def _compute_phonological_complexity(
        self,
        features_list: list[PhonologicalFeatures]
    ) -> float:
        """
        Compute phonological complexity score.

        Based on:
        - Syllable count
        - Consonant clusters
        - CV ratio
        """
        if not features_list:
            return 0.0

        complexity_scores = []
        for features in features_list:
            score = 0.0

            # More syllables = more complex
            score += min(features.syllable_count / 5.0, 0.4)

            # Consonant clusters add complexity
            score += min(len(features.consonant_clusters) * 0.2, 0.3)

            # Extreme CV ratios are complex
            cv_ratio = features.consonant_count / features.vowel_count if features.vowel_count > 0 else 0
            if cv_ratio > 2.0 or cv_ratio < 0.5:
                score += 0.3

            complexity_scores.append(min(score, 1.0))  # Cap at 1.0

        return sum(complexity_scores) / len(complexity_scores)

