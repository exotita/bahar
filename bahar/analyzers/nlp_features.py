"""
NLP Features Module using spaCy.

Provides advanced NLP analysis including:
- Part-of-Speech (POS) tagging
- Named Entity Recognition (NER)
- Dependency parsing
- Sentence structure analysis
- Readability metrics
- Syntactic complexity
"""

from __future__ import annotations

from collections import Counter
from typing import Any

try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Doc = Any  # type: ignore


def check_and_download_model(model_name: str) -> bool:
    """
    Check if spaCy model is available, download if not.

    Args:
        model_name: Name of the spaCy model

    Returns:
        True if model is available or successfully downloaded, False otherwise
    """
    if not SPACY_AVAILABLE:
        return False

    try:
        # Try to load the model
        spacy.load(model_name)
        return True
    except OSError:
        # Model not found, try to download
        try:
            import subprocess
            import sys

            print(f"📥 Downloading spaCy model '{model_name}'...")
            print(f"This may take a few moments...")

            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Verify the model loads
                spacy.load(model_name)
                print(f"✓ Model '{model_name}' downloaded and loaded successfully!")
                return True
            else:
                print(f"✗ Failed to download model: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"✗ Download timeout for model '{model_name}'")
            return False
        except Exception as e:
            print(f"✗ Error downloading model '{model_name}': {e}")
            return False


class NLPFeatures:
    """Container for NLP analysis results."""

    def __init__(
        self,
        text: str,
        language: str,
        # Token analysis
        num_tokens: int,
        num_sentences: int,
        avg_token_length: float,
        avg_sentence_length: float,
        # POS distribution
        pos_distribution: dict[str, int],
        # Named entities
        entities: list[tuple[str, str, str]],  # (text, label, explanation)
        entity_types: dict[str, int],
        # Dependency analysis
        dependency_types: dict[str, int],
        # Syntactic features
        num_nouns: int,
        num_verbs: int,
        num_adjectives: int,
        num_adverbs: int,
        noun_verb_ratio: float,
        # Readability
        lexical_diversity: float,  # unique words / total words
        # Advanced features
        has_negation: bool,
        question_count: int,
        exclamation_count: int,
    ) -> None:
        self.text = text
        self.language = language

        # Token analysis
        self.num_tokens = num_tokens
        self.num_sentences = num_sentences
        self.avg_token_length = avg_token_length
        self.avg_sentence_length = avg_sentence_length

        # POS distribution
        self.pos_distribution = pos_distribution

        # Named entities
        self.entities = entities
        self.entity_types = entity_types

        # Dependency analysis
        self.dependency_types = dependency_types

        # Syntactic features
        self.num_nouns = num_nouns
        self.num_verbs = num_verbs
        self.num_adjectives = num_adjectives
        self.num_adverbs = num_adverbs
        self.noun_verb_ratio = noun_verb_ratio

        # Readability
        self.lexical_diversity = lexical_diversity

        # Advanced features
        self.has_negation = has_negation
        self.question_count = question_count
        self.exclamation_count = exclamation_count

    def get_summary(self) -> dict[str, Any]:
        """Get a summary dictionary of all features."""
        return {
            "text_stats": {
                "tokens": self.num_tokens,
                "sentences": self.num_sentences,
                "avg_token_length": round(self.avg_token_length, 2),
                "avg_sentence_length": round(self.avg_sentence_length, 2),
            },
            "pos_distribution": self.pos_distribution,
            "entities": {
                "count": len(self.entities),
                "types": self.entity_types,
                "list": self.entities[:10],  # Top 10
            },
            "syntactic": {
                "nouns": self.num_nouns,
                "verbs": self.num_verbs,
                "adjectives": self.num_adjectives,
                "adverbs": self.num_adverbs,
                "noun_verb_ratio": round(self.noun_verb_ratio, 2),
            },
            "readability": {
                "lexical_diversity": round(self.lexical_diversity, 2),
            },
            "features": {
                "has_negation": self.has_negation,
                "questions": self.question_count,
                "exclamations": self.exclamation_count,
            },
        }

    def __repr__(self) -> str:
        return (
            f"NLPFeatures("
            f"tokens={self.num_tokens}, "
            f"sentences={self.num_sentences}, "
            f"entities={len(self.entities)}, "
            f"pos_types={len(self.pos_distribution)})"
        )


class NLPAnalyzer:
    """
    Advanced NLP analyzer using spaCy.

    Provides comprehensive linguistic and syntactic analysis.
    """

    def __init__(self, language: str = "english") -> None:
        """
        Initialize NLP analyzer.

        Args:
            language: Language code (english, dutch, persian)
        """
        if not SPACY_AVAILABLE:
            raise RuntimeError(
                "spaCy is not installed. Install with: pip install spacy\n"
                "Then download language model: python -m spacy download en_core_web_sm"
            )

        self.language = language
        self._nlp: Any = None
        self._model_name = self._get_model_name(language)

    def _get_model_name(self, language: str) -> str:
        """Get spaCy model name for language."""
        model_map = {
            "english": "en_core_web_sm",
            "dutch": "nl_core_news_sm",
            "persian": "en_core_web_sm",  # Fallback to English for Persian
        }
        return model_map.get(language, "en_core_web_sm")

    def load_model(self, auto_download: bool = True) -> None:
        """
        Load spaCy language model.

        Args:
            auto_download: Automatically download model if not found
        """
        if auto_download:
            # Check and download if needed
            success = check_and_download_model(self._model_name)
            if not success:
                raise RuntimeError(
                    f"Failed to load or download spaCy model '{self._model_name}'.\n"
                    f"Please install manually: python -m spacy download {self._model_name}"
                )
            # Load the model
            self._nlp = spacy.load(self._model_name)
        else:
            # Just try to load without downloading
            try:
                self._nlp = spacy.load(self._model_name)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{self._model_name}' not found.\n"
                    f"Download it with: python -m spacy download {self._model_name}"
                )

    def analyze(self, text: str) -> NLPFeatures:
        """
        Perform comprehensive NLP analysis.

        Args:
            text: Input text to analyze

        Returns:
            NLPFeatures object with all analysis results
        """
        if self._nlp is None:
            self.load_model()

        # Process text
        doc: Doc = self._nlp(text)

        # Token analysis
        tokens = [token for token in doc if not token.is_space]
        num_tokens = len(tokens)
        num_sentences = len(list(doc.sents))

        avg_token_length = (
            sum(len(token.text) for token in tokens) / num_tokens
            if num_tokens > 0 else 0
        )

        avg_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0

        # POS distribution
        pos_counts = Counter(token.pos_ for token in tokens)
        pos_distribution = dict(pos_counts.most_common())

        # Named entities
        entities = [
            (ent.text, ent.label_, spacy.explain(ent.label_) or ent.label_)
            for ent in doc.ents
        ]
        entity_types = dict(Counter(ent.label_ for ent in doc.ents))

        # Dependency analysis
        dependency_types = dict(Counter(token.dep_ for token in tokens))

        # Syntactic features
        num_nouns = sum(1 for token in tokens if token.pos_ in ["NOUN", "PROPN"])
        num_verbs = sum(1 for token in tokens if token.pos_ == "VERB")
        num_adjectives = sum(1 for token in tokens if token.pos_ == "ADJ")
        num_adverbs = sum(1 for token in tokens if token.pos_ == "ADV")

        noun_verb_ratio = num_nouns / num_verbs if num_verbs > 0 else num_nouns

        # Readability - lexical diversity
        unique_lemmas = len(set(token.lemma_.lower() for token in tokens if token.is_alpha))
        lexical_diversity = unique_lemmas / num_tokens if num_tokens > 0 else 0

        # Advanced features
        has_negation = any(token.dep_ == "neg" for token in tokens)
        question_count = text.count("?")
        exclamation_count = text.count("!")

        return NLPFeatures(
            text=text,
            language=self.language,
            num_tokens=num_tokens,
            num_sentences=num_sentences,
            avg_token_length=avg_token_length,
            avg_sentence_length=avg_sentence_length,
            pos_distribution=pos_distribution,
            entities=entities,
            entity_types=entity_types,
            dependency_types=dependency_types,
            num_nouns=num_nouns,
            num_verbs=num_verbs,
            num_adjectives=num_adjectives,
            num_adverbs=num_adverbs,
            noun_verb_ratio=noun_verb_ratio,
            lexical_diversity=lexical_diversity,
            has_negation=has_negation,
            question_count=question_count,
            exclamation_count=exclamation_count,
        )

    def analyze_batch(self, texts: list[str]) -> list[NLPFeatures]:
        """
        Analyze multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of NLPFeatures objects
        """
        return [self.analyze(text) for text in texts]


def format_nlp_analysis(features: NLPFeatures) -> str:
    """Format NLP features for display."""
    lines: list[str] = []
    lines.append("\nNLP Analysis:")
    lines.append("-" * 60)

    # Text statistics
    lines.append(f"  Tokens:        {features.num_tokens}")
    lines.append(f"  Sentences:     {features.num_sentences}")
    lines.append(f"  Avg Token Len: {features.avg_token_length:.2f}")
    lines.append(f"  Avg Sent Len:  {features.avg_sentence_length:.2f}")
    lines.append("")

    # POS distribution (top 5)
    lines.append("  POS Distribution:")
    for pos, count in list(features.pos_distribution.items())[:5]:
        lines.append(f"    {pos:10s}: {count}")
    lines.append("")

    # Named entities
    if features.entities:
        lines.append(f"  Named Entities ({len(features.entities)}):")
        for text, label, explanation in features.entities[:5]:
            lines.append(f"    {text:20s} [{label}] - {explanation}")
    lines.append("")

    # Syntactic features
    lines.append("  Syntactic Features:")
    lines.append(f"    Nouns:      {features.num_nouns}")
    lines.append(f"    Verbs:      {features.num_verbs}")
    lines.append(f"    Adjectives: {features.num_adjectives}")
    lines.append(f"    Adverbs:    {features.num_adverbs}")
    lines.append(f"    N/V Ratio:  {features.noun_verb_ratio:.2f}")
    lines.append("")

    # Readability
    lines.append("  Readability:")
    lines.append(f"    Lexical Diversity: {features.lexical_diversity:.2%}")
    lines.append("")

    # Features
    lines.append("  Features:")
    lines.append(f"    Negation:     {'Yes' if features.has_negation else 'No'}")
    lines.append(f"    Questions:    {features.question_count}")
    lines.append(f"    Exclamations: {features.exclamation_count}")

    return "\n".join(lines)

