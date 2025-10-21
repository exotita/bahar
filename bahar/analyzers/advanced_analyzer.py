"""
Advanced Linguistic Analyzer - Unified Interface.

This module provides a unified interface for all advanced linguistic analysis:
- Lexical & Compositional Semantics
- Morphology & Phonology
- Distributional Semantics & Embeddings
- Pragmatics & Discourse

Academic research focused with comprehensive statistics and export capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bahar.analyzers.semantic_analyzer import SemanticAnalyzer, SemanticFeatures
from bahar.analyzers.morphology_analyzer import MorphologyAnalyzer, MorphologyFeatures
from bahar.analyzers.embedding_analyzer import EmbeddingAnalyzer, EmbeddingFeatures
from bahar.analyzers.discourse_analyzer import DiscourseAnalyzer, DiscourseFeatures


@dataclass
class AdvancedAnalysisResult:
    """
    Comprehensive advanced linguistic analysis result.

    Combines results from all analyzers with unified export capabilities.
    """

    text: str
    semantic_features: SemanticFeatures | None = None
    morphology_features: MorphologyFeatures | None = None
    embedding_features: EmbeddingFeatures | None = None
    discourse_features: DiscourseFeatures | None = None

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of all analyses."""
        summary: dict[str, Any] = {
            "text": self.text,
            "text_length": len(self.text),
            "word_count": len(self.text.split()),
        }

        if self.semantic_features:
            summary["semantic"] = self.semantic_features.get_summary()

        if self.morphology_features:
            summary["morphology"] = self.morphology_features.get_summary()

        if self.embedding_features:
            summary["embeddings"] = self.embedding_features.get_summary()

        if self.discourse_features:
            summary["discourse"] = self.discourse_features.get_summary()

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        result: dict[str, Any] = {
            "text": self.text,
            "analyses": {},
        }

        if self.semantic_features:
            result["analyses"]["semantic"] = self.semantic_features.to_dict()

        if self.morphology_features:
            result["analyses"]["morphology"] = self.morphology_features.to_dict()

        if self.embedding_features:
            result["analyses"]["embeddings"] = self.embedding_features.to_dict()

        if self.discourse_features:
            result["analyses"]["discourse"] = self.discourse_features.to_dict()

        return result

    def export_academic_format(self) -> dict[str, Any]:
        """
        Export in academic research format.

        Flattened structure suitable for CSV export and statistical analysis.
        """
        export_data: dict[str, Any] = {
            "text": self.text,
            "text_length": len(self.text),
            "word_count": len(self.text.split()),
        }

        # Semantic features
        if self.semantic_features:
            export_data["semantic_lexical_diversity"] = self.semantic_features.lexical_diversity
            export_data["semantic_density"] = self.semantic_features.semantic_density
            export_data["semantic_polysemy_rate"] = self.semantic_features.polysemy_rate
            export_data["semantic_cohesion_score"] = self.semantic_features.cohesion_score
            export_data["semantic_word_senses_count"] = len(self.semantic_features.word_senses)
            export_data["semantic_lexical_chains_count"] = len(self.semantic_features.lexical_chains)

        # Morphology features
        if self.morphology_features:
            export_data["morphology_morphemes_per_word"] = self.morphology_features.morphemes_per_word
            export_data["morphology_complexity"] = self.morphology_features.morphological_complexity
            export_data["morphology_derivational_ratio"] = self.morphology_features.derivational_ratio
            export_data["morphology_inflectional_ratio"] = self.morphology_features.inflectional_ratio
            export_data["morphology_syllables_per_word"] = self.morphology_features.syllables_per_word
            export_data["morphology_cv_ratio"] = self.morphology_features.consonant_vowel_ratio
            export_data["morphology_phonological_complexity"] = self.morphology_features.phonological_complexity

        # Embedding features
        if self.embedding_features:
            export_data["embedding_dimensionality"] = self.embedding_features.vector_dimensionality
            export_data["embedding_avg_vector_norm"] = self.embedding_features.avg_vector_norm
            export_data["embedding_semantic_density"] = self.embedding_features.semantic_density
            export_data["embedding_cluster_quality"] = self.embedding_features.cluster_quality
            export_data["embedding_effective_dimensions"] = self.embedding_features.effective_dimensions
            export_data["embedding_num_clusters"] = len(self.embedding_features.clusters)

        # Discourse features
        if self.discourse_features:
            export_data["discourse_entity_density"] = self.discourse_features.entity_density
            export_data["discourse_avg_chain_length"] = self.discourse_features.avg_chain_length
            export_data["discourse_topic_continuity"] = self.discourse_features.topic_continuity
            export_data["discourse_coherence_score"] = self.discourse_features.coherence_score
            export_data["discourse_num_entities"] = len(self.discourse_features.entity_mentions)
            export_data["discourse_num_chains"] = len(self.discourse_features.coreference_chains)

        return export_data

    def __repr__(self) -> str:
        """String representation."""
        analyses = []
        if self.semantic_features:
            analyses.append("semantic")
        if self.morphology_features:
            analyses.append("morphology")
        if self.embedding_features:
            analyses.append("embeddings")
        if self.discourse_features:
            analyses.append("discourse")

        return (
            f"AdvancedAnalysisResult(\n"
            f"  text='{self.text[:50]}...',\n"
            f"  analyses={analyses}\n"
            f")"
        )


class AdvancedLinguisticAnalyzer:
    """
    Unified analyzer for advanced linguistic features.

    Combines:
    - Lexical & Compositional Semantics (WordNet, WSD, semantic similarity)
    - Morphology & Phonology (morphemes, syllables, complexity)
    - Distributional Semantics & Embeddings (Word2Vec, semantic space)
    - Pragmatics & Discourse (coherence, coreference, information flow)

    Academic research focused with comprehensive statistics and export capabilities.

    Example:
        >>> analyzer = AdvancedLinguisticAnalyzer(
        ...     language="english",
        ...     enable_semantics=True,
        ...     enable_morphology=True,
        ...     enable_embeddings=True,
        ...     enable_discourse=True,
        ... )
        >>> analyzer.load_models()
        >>> result = analyzer.analyze("Your text here")
        >>> print(result.get_summary())
        >>> data = result.export_academic_format()
    """

    def __init__(
        self,
        language: str = "english",
        enable_semantics: bool = True,
        enable_morphology: bool = True,
        enable_embeddings: bool = True,
        enable_discourse: bool = True,
    ) -> None:
        """
        Initialize advanced linguistic analyzer.

        Args:
            language: Language code (english, dutch, persian)
            enable_semantics: Enable lexical & compositional semantics analysis
            enable_morphology: Enable morphology & phonology analysis
            enable_embeddings: Enable distributional semantics & embeddings analysis
            enable_discourse: Enable pragmatics & discourse analysis
        """
        self.language = language
        self.enable_semantics = enable_semantics
        self.enable_morphology = enable_morphology
        self.enable_embeddings = enable_embeddings
        self.enable_discourse = enable_discourse

        # Initialize analyzers
        self.semantic_analyzer: SemanticAnalyzer | None = None
        self.morphology_analyzer: MorphologyAnalyzer | None = None
        self.embedding_analyzer: EmbeddingAnalyzer | None = None
        self.discourse_analyzer: DiscourseAnalyzer | None = None

        if enable_semantics:
            self.semantic_analyzer = SemanticAnalyzer()

        if enable_morphology:
            self.morphology_analyzer = MorphologyAnalyzer(language=language)

        if enable_embeddings:
            self.embedding_analyzer = EmbeddingAnalyzer(language=language)

        if enable_discourse:
            self.discourse_analyzer = DiscourseAnalyzer(language=language)

        self._loaded = False

    def load_models(self) -> None:
        """Load all enabled analyzer models."""
        if self._loaded:
            return

        if self.semantic_analyzer:
            self.semantic_analyzer.load_model()

        if self.morphology_analyzer:
            self.morphology_analyzer.load_model()

        if self.embedding_analyzer:
            self.embedding_analyzer.load_model()

        if self.discourse_analyzer:
            self.discourse_analyzer.load_model()

        self._loaded = True

    def analyze(self, text: str, **kwargs: Any) -> AdvancedAnalysisResult:
        """
        Perform comprehensive advanced linguistic analysis.

        Args:
            text: Input text to analyze
            **kwargs: Additional parameters for specific analyzers
                - top_k_neighbors: For embedding analyzer (default: 5)
                - n_clusters: For embedding analyzer (default: 3)

        Returns:
            AdvancedAnalysisResult with all enabled analyses
        """
        if not self._loaded:
            self.load_models()

        result = AdvancedAnalysisResult(text=text)

        # Semantic analysis
        if self.semantic_analyzer:
            result.semantic_features = self.semantic_analyzer.analyze(text)

        # Morphology analysis
        if self.morphology_analyzer:
            result.morphology_features = self.morphology_analyzer.analyze(text)

        # Embedding analysis
        if self.embedding_analyzer:
            top_k = kwargs.get('top_k_neighbors', 5)
            n_clusters = kwargs.get('n_clusters', 3)
            result.embedding_features = self.embedding_analyzer.analyze(
                text,
                top_k_neighbors=top_k,
                n_clusters=n_clusters
            )

        # Discourse analysis
        if self.discourse_analyzer:
            result.discourse_features = self.discourse_analyzer.analyze(text)

        return result

    def analyze_batch(
        self,
        texts: list[str],
        **kwargs: Any
    ) -> list[AdvancedAnalysisResult]:
        """
        Analyze multiple texts.

        Args:
            texts: List of input texts
            **kwargs: Additional parameters for analyzers

        Returns:
            List of AdvancedAnalysisResult objects
        """
        return [self.analyze(text, **kwargs) for text in texts]

    def get_capabilities(self) -> dict[str, bool]:
        """Get enabled capabilities."""
        return {
            "semantics": self.enable_semantics,
            "morphology": self.enable_morphology,
            "embeddings": self.enable_embeddings,
            "discourse": self.enable_discourse,
        }

    def __repr__(self) -> str:
        """String representation."""
        capabilities = self.get_capabilities()
        enabled = [k for k, v in capabilities.items() if v]
        return (
            f"AdvancedLinguisticAnalyzer(\n"
            f"  language='{self.language}',\n"
            f"  enabled={enabled}\n"
            f")"
        )


def export_to_csv(results: list[AdvancedAnalysisResult], filepath: str) -> None:
    """
    Export analysis results to CSV file.

    Args:
        results: List of analysis results
        filepath: Output CSV file path
    """
    import csv

    if not results:
        return

    # Get all keys from first result
    sample_data = results[0].export_academic_format()
    fieldnames = list(sample_data.keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.export_academic_format())


def export_to_json(results: list[AdvancedAnalysisResult], filepath: str) -> None:
    """
    Export analysis results to JSON file.

    Args:
        results: List of analysis results
        filepath: Output JSON file path
    """
    import json

    data = [result.to_dict() for result in results]

    with open(filepath, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=2, ensure_ascii=False)

