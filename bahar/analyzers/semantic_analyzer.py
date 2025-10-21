"""
Semantic Analyzer for Lexical and Compositional Semantics.

This module provides comprehensive semantic analysis including:
- Word sense disambiguation (WSD)
- Semantic similarity and relatedness
- Semantic roles and frame semantics
- Lexical chains and cohesion analysis

Academic research focused with statistical metrics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import wordnet_ic


@dataclass
class WordSense:
    """Word sense from disambiguation."""

    word: str
    pos: str  # Part of speech
    synset: Any  # WordNet synset
    definition: str
    examples: list[str]
    confidence: float
    hypernyms: list[str] = field(default_factory=list)
    hyponyms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "word": self.word,
            "pos": self.pos,
            "synset": str(self.synset),
            "definition": self.definition,
            "examples": self.examples,
            "confidence": self.confidence,
            "hypernyms": self.hypernyms,
            "hyponyms": self.hyponyms,
        }


@dataclass
class SemanticSimilarity:
    """Semantic similarity between two words."""

    word1: str
    word2: str
    wu_palmer: float | None = None
    path_similarity: float | None = None
    leacock_chodorow: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "word1": self.word1,
            "word2": self.word2,
            "wu_palmer": self.wu_palmer,
            "path_similarity": self.path_similarity,
            "leacock_chodorow": self.leacock_chodorow,
        }


@dataclass
class LexicalChain:
    """Lexical chain showing semantic continuity."""

    words: list[str]
    chain_type: str  # repetition, synonym, hypernym, etc.
    strength: float  # Cohesion score
    positions: list[int]  # Word positions in text

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "words": self.words,
            "chain_type": self.chain_type,
            "strength": self.strength,
            "positions": self.positions,
            "length": len(self.words),
        }


@dataclass
class SemanticFeatures:
    """Complete semantic analysis results."""

    text: str
    word_senses: list[WordSense] = field(default_factory=list)
    similarities: list[SemanticSimilarity] = field(default_factory=list)
    lexical_chains: list[LexicalChain] = field(default_factory=list)

    # Statistics
    lexical_diversity: float = 0.0  # Type-Token Ratio
    semantic_density: float = 0.0
    polysemy_rate: float = 0.0
    cohesion_score: float = 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "text_length": len(self.text),
            "word_count": len(self.text.split()),
            "disambiguated_words": len(self.word_senses),
            "lexical_chains": len(self.lexical_chains),
            "lexical_diversity": self.lexical_diversity,
            "semantic_density": self.semantic_density,
            "polysemy_rate": self.polysemy_rate,
            "cohesion_score": self.cohesion_score,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "text": self.text,
            "word_senses": [ws.to_dict() for ws in self.word_senses],
            "similarities": [sim.to_dict() for sim in self.similarities],
            "lexical_chains": [chain.to_dict() for chain in self.lexical_chains],
            "statistics": {
                "lexical_diversity": self.lexical_diversity,
                "semantic_density": self.semantic_density,
                "polysemy_rate": self.polysemy_rate,
                "cohesion_score": self.cohesion_score,
            }
        }


class SemanticAnalyzer:
    """
    Analyzer for lexical and compositional semantics.

    Provides:
    - Word sense disambiguation using Lesk algorithm
    - Semantic similarity metrics (Wu-Palmer, Path, Leacock-Chodorow)
    - Lexical chain construction
    - Semantic density and cohesion metrics

    Academic research focused with comprehensive statistics.
    """

    def __init__(self) -> None:
        """Initialize semantic analyzer."""
        self._loaded = False
        self._brown_ic: Any = None

    def load_model(self) -> None:
        """Load required NLTK data."""
        if self._loaded:
            return

        # Ensure WordNet data is available
        try:
            wn.ensure_loaded()
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        # Load information content for Leacock-Chodorow
        try:
            self._brown_ic = wordnet_ic.ic('ic-brown.dat')
        except LookupError:
            nltk.download('wordnet_ic', quiet=True)
            self._brown_ic = wordnet_ic.ic('ic-brown.dat')

        self._loaded = True

    def analyze(self, text: str) -> SemanticFeatures:
        """
        Perform comprehensive semantic analysis.

        Args:
            text: Input text to analyze

        Returns:
            SemanticFeatures with complete analysis
        """
        if not self._loaded:
            self.load_model()

        # Tokenize
        from nltk.tokenize import word_tokenize
        from nltk import pos_tag

        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)

        # Filter to content words (nouns, verbs, adjectives, adverbs)
        content_words = [
            (word, pos) for word, pos in pos_tags
            if pos.startswith(('NN', 'VB', 'JJ', 'RB')) and len(word) > 2
        ]

        # Word sense disambiguation
        word_senses = self._disambiguate_words(text, content_words)

        # Compute semantic similarities
        similarities = self._compute_similarities(content_words[:10])  # Limit for performance

        # Build lexical chains
        lexical_chains = self._build_lexical_chains(content_words)

        # Compute statistics
        lexical_diversity = self._compute_lexical_diversity(tokens)
        semantic_density = self._compute_semantic_density(content_words, len(tokens))
        polysemy_rate = self._compute_polysemy_rate(content_words)
        cohesion_score = self._compute_cohesion_score(lexical_chains, len(content_words))

        return SemanticFeatures(
            text=text,
            word_senses=word_senses,
            similarities=similarities,
            lexical_chains=lexical_chains,
            lexical_diversity=lexical_diversity,
            semantic_density=semantic_density,
            polysemy_rate=polysemy_rate,
            cohesion_score=cohesion_score,
        )

    def _disambiguate_words(
        self,
        text: str,
        content_words: list[tuple[str, str]]
    ) -> list[WordSense]:
        """Disambiguate word senses using Lesk algorithm."""
        word_senses = []

        for word, pos in content_words[:20]:  # Limit for performance
            # Convert POS tag to WordNet format
            wn_pos = self._get_wordnet_pos(pos)
            if not wn_pos:
                continue

            # Get all possible synsets
            synsets = wn.synsets(word, pos=wn_pos)
            if not synsets:
                continue

            # Use Lesk algorithm for disambiguation
            try:
                best_synset = lesk(text.split(), word, wn_pos)
                if not best_synset:
                    best_synset = synsets[0]  # Default to first sense
            except Exception:
                best_synset = synsets[0]

            # Calculate confidence (inverse of number of senses)
            confidence = 1.0 / len(synsets) if len(synsets) > 1 else 1.0

            # Get hypernyms and hyponyms
            hypernyms = [h.name() for h in best_synset.hypernyms()[:3]]
            hyponyms = [h.name() for h in best_synset.hyponyms()[:3]]

            word_sense = WordSense(
                word=word,
                pos=pos,
                synset=best_synset,
                definition=best_synset.definition(),
                examples=best_synset.examples()[:2],
                confidence=confidence,
                hypernyms=hypernyms,
                hyponyms=hyponyms,
            )
            word_senses.append(word_sense)

        return word_senses

    def _compute_similarities(
        self,
        content_words: list[tuple[str, str]]
    ) -> list[SemanticSimilarity]:
        """Compute semantic similarities between word pairs."""
        similarities = []

        # Compare first few words for performance
        for i in range(min(5, len(content_words))):
            for j in range(i + 1, min(i + 3, len(content_words))):
                word1, pos1 = content_words[i]
                word2, pos2 = content_words[j]

                wn_pos1 = self._get_wordnet_pos(pos1)
                wn_pos2 = self._get_wordnet_pos(pos2)

                if not wn_pos1 or not wn_pos2:
                    continue

                synsets1 = wn.synsets(word1, pos=wn_pos1)
                synsets2 = wn.synsets(word2, pos=wn_pos2)

                if not synsets1 or not synsets2:
                    continue

                # Use first synset for each word
                syn1 = synsets1[0]
                syn2 = synsets2[0]

                # Compute similarities
                wu_palmer = syn1.wup_similarity(syn2)
                path_sim = syn1.path_similarity(syn2)

                # Leacock-Chodorow requires same POS
                lch = None
                if wn_pos1 == wn_pos2 and self._brown_ic:
                    try:
                        lch = syn1.lch_similarity(syn2, self._brown_ic)
                    except Exception:
                        pass

                similarity = SemanticSimilarity(
                    word1=word1,
                    word2=word2,
                    wu_palmer=wu_palmer,
                    path_similarity=path_sim,
                    leacock_chodorow=lch,
                )
                similarities.append(similarity)

        return similarities

    def _build_lexical_chains(
        self,
        content_words: list[tuple[str, str]]
    ) -> list[LexicalChain]:
        """Build lexical chains showing semantic continuity."""
        chains = []
        word_positions = {}

        # Track word positions
        for i, (word, pos) in enumerate(content_words):
            if word not in word_positions:
                word_positions[word] = []
            word_positions[word].append(i)

        # Find repetition chains
        for word, positions in word_positions.items():
            if len(positions) > 1:
                chain = LexicalChain(
                    words=[word] * len(positions),
                    chain_type="repetition",
                    strength=len(positions) / len(content_words),
                    positions=positions,
                )
                chains.append(chain)

        # Find synonym chains (simplified)
        processed = set()
        for i, (word1, pos1) in enumerate(content_words):
            if word1 in processed:
                continue

            wn_pos1 = self._get_wordnet_pos(pos1)
            if not wn_pos1:
                continue

            synsets1 = wn.synsets(word1, pos=wn_pos1)
            if not synsets1:
                continue

            chain_words = [word1]
            chain_positions = [i]

            for j in range(i + 1, min(i + 10, len(content_words))):
                word2, pos2 = content_words[j]
                if word2 in processed:
                    continue

                wn_pos2 = self._get_wordnet_pos(pos2)
                if not wn_pos2:
                    continue

                synsets2 = wn.synsets(word2, pos=wn_pos2)
                if not synsets2:
                    continue

                # Check if words are synonyms
                if synsets1[0].wup_similarity(synsets2[0]) and synsets1[0].wup_similarity(synsets2[0]) > 0.8:
                    chain_words.append(word2)
                    chain_positions.append(j)
                    processed.add(word2)

            if len(chain_words) > 1:
                chain = LexicalChain(
                    words=chain_words,
                    chain_type="synonym",
                    strength=len(chain_words) / len(content_words),
                    positions=chain_positions,
                )
                chains.append(chain)

            processed.add(word1)

        return chains

    def _compute_lexical_diversity(self, tokens: list[str]) -> float:
        """Compute Type-Token Ratio (TTR)."""
        if not tokens:
            return 0.0
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)

    def _compute_semantic_density(
        self,
        content_words: list[tuple[str, str]],
        total_tokens: int
    ) -> float:
        """Compute semantic density (content words / total words)."""
        if total_tokens == 0:
            return 0.0
        return len(content_words) / total_tokens

    def _compute_polysemy_rate(self, content_words: list[tuple[str, str]]) -> float:
        """Compute average polysemy rate (senses per word)."""
        if not content_words:
            return 0.0

        total_senses = 0
        count = 0

        for word, pos in content_words:
            wn_pos = self._get_wordnet_pos(pos)
            if not wn_pos:
                continue

            synsets = wn.synsets(word, pos=wn_pos)
            if synsets:
                total_senses += len(synsets)
                count += 1

        return total_senses / count if count > 0 else 0.0

    def _compute_cohesion_score(
        self,
        lexical_chains: list[LexicalChain],
        content_word_count: int
    ) -> float:
        """Compute overall cohesion score from lexical chains."""
        if not lexical_chains or content_word_count == 0:
            return 0.0

        # Average chain strength weighted by chain length
        total_strength = sum(chain.strength * len(chain.words) for chain in lexical_chains)
        total_words_in_chains = sum(len(chain.words) for chain in lexical_chains)

        if total_words_in_chains == 0:
            return 0.0

        return total_strength / total_words_in_chains

    def _get_wordnet_pos(self, treebank_tag: str) -> str | None:
        """Convert Penn Treebank POS tag to WordNet POS tag."""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

