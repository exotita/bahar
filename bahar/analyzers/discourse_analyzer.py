"""
Discourse Analyzer for Pragmatics and Discourse Analysis.

This module provides discourse and pragmatic analysis:
- Text coherence metrics
- Information flow tracking
- Entity continuity analysis
- Discourse structure analysis
- Basic coreference tracking

Academic research focused with statistical metrics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import spacy
from spacy.tokens import Doc
import networkx as nx


@dataclass
class EntityMention:
    """Entity mention in text."""

    text: str
    label: str  # Entity type (PERSON, ORG, etc.)
    start_pos: int
    end_pos: int
    sentence_id: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "text": self.text,
            "label": self.label,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "sentence_id": self.sentence_id,
        }


@dataclass
class CoreferenceChain:
    """Coreference chain (simplified)."""

    chain_id: int
    mentions: list[EntityMention]
    chain_type: str  # pronoun, nominal, proper

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "chain_id": self.chain_id,
            "mentions": [m.to_dict() for m in self.mentions],
            "chain_length": len(self.mentions),
            "chain_type": self.chain_type,
        }


@dataclass
class DiscourseRelation:
    """Discourse relation between text segments."""

    relation_type: str  # cause, contrast, elaboration, etc.
    segment1: str
    segment2: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "relation_type": self.relation_type,
            "segment1": segment1[:100],  # Truncate for export
            "segment2": segment2[:100],
            "confidence": self.confidence,
        }


@dataclass
class InformationFlow:
    """Information flow analysis."""

    sentence_id: int
    new_entities: list[str]
    given_entities: list[str]
    topic: str | None
    focus: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "sentence_id": self.sentence_id,
            "new_entities": self.new_entities,
            "given_entities": self.given_entities,
            "topic": self.topic,
            "focus": self.focus,
        }


@dataclass
class DiscourseFeatures:
    """Complete discourse analysis results."""

    text: str
    entity_mentions: list[EntityMention] = field(default_factory=list)
    coreference_chains: list[CoreferenceChain] = field(default_factory=list)
    discourse_relations: list[DiscourseRelation] = field(default_factory=list)
    information_flow: list[InformationFlow] = field(default_factory=list)

    # Statistics
    entity_density: float = 0.0  # Entities per sentence
    avg_chain_length: float = 0.0
    topic_continuity: float = 0.0
    coherence_score: float = 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "text_length": len(self.text),
            "num_sentences": len(self.information_flow),
            "num_entities": len(self.entity_mentions),
            "num_chains": len(self.coreference_chains),
            "entity_density": self.entity_density,
            "avg_chain_length": self.avg_chain_length,
            "topic_continuity": self.topic_continuity,
            "coherence_score": self.coherence_score,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "text": self.text,
            "entity_mentions": [em.to_dict() for em in self.entity_mentions],
            "coreference_chains": [cc.to_dict() for cc in self.coreference_chains],
            "discourse_relations": [dr.to_dict() for dr in self.discourse_relations],
            "information_flow": [inf.to_dict() for inf in self.information_flow],
            "statistics": {
                "entity_density": self.entity_density,
                "avg_chain_length": self.avg_chain_length,
                "topic_continuity": self.topic_continuity,
                "coherence_score": self.coherence_score,
            }
        }


class DiscourseAnalyzer:
    """
    Analyzer for discourse and pragmatics.

    Provides:
    - Entity mention tracking
    - Basic coreference resolution
    - Information flow analysis
    - Discourse coherence metrics
    - Entity continuity tracking

    Note: Advanced coreference requires AllenNLP (not available in Python 3.12).
    This implementation uses rule-based approaches with spaCy.

    Academic research focused with comprehensive statistics.
    """

    # Discourse connectives by type
    CAUSAL_CONNECTIVES = {
        'because', 'since', 'as', 'therefore', 'thus', 'hence', 'consequently',
        'so', 'accordingly', 'for this reason'
    }

    CONTRAST_CONNECTIVES = {
        'but', 'however', 'although', 'though', 'yet', 'nevertheless',
        'nonetheless', 'on the other hand', 'in contrast', 'whereas'
    }

    ELABORATION_CONNECTIVES = {
        'also', 'furthermore', 'moreover', 'additionally', 'in addition',
        'besides', 'for example', 'for instance', 'specifically'
    }

    TEMPORAL_CONNECTIVES = {
        'then', 'next', 'after', 'before', 'while', 'when', 'meanwhile',
        'subsequently', 'previously', 'finally'
    }

    def __init__(self, language: str = "english") -> None:
        """
        Initialize discourse analyzer.

        Args:
            language: Language code (english, dutch, persian)
        """
        self.language = language
        self._nlp: Any = None
        self._loaded = False

    def load_model(self) -> None:
        """Load spaCy model with NER."""
        if self._loaded:
            return

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

        self._loaded = True

    def analyze(self, text: str) -> DiscourseFeatures:
        """
        Perform comprehensive discourse analysis.

        Args:
            text: Input text to analyze

        Returns:
            DiscourseFeatures with complete analysis
        """
        if not self._loaded:
            self.load_model()

        # Process text
        doc = self._nlp(text)

        # Extract entity mentions
        entity_mentions = self._extract_entity_mentions(doc)

        # Build coreference chains (simplified)
        coreference_chains = self._build_coreference_chains(doc, entity_mentions)

        # Detect discourse relations
        discourse_relations = self._detect_discourse_relations(doc)

        # Analyze information flow
        information_flow = self._analyze_information_flow(doc, entity_mentions)

        # Compute statistics
        entity_density = self._compute_entity_density(entity_mentions, doc)
        avg_chain_length = self._compute_avg_chain_length(coreference_chains)
        topic_continuity = self._compute_topic_continuity(information_flow)
        coherence_score = self._compute_coherence_score(
            entity_mentions, coreference_chains, information_flow
        )

        return DiscourseFeatures(
            text=text,
            entity_mentions=entity_mentions,
            coreference_chains=coreference_chains,
            discourse_relations=discourse_relations,
            information_flow=information_flow,
            entity_density=entity_density,
            avg_chain_length=avg_chain_length,
            topic_continuity=topic_continuity,
            coherence_score=coherence_score,
        )

    def _extract_entity_mentions(self, doc: Doc) -> list[EntityMention]:
        """Extract entity mentions using spaCy NER."""
        mentions = []

        # Get sentence boundaries
        sentences = list(doc.sents)
        sent_starts = {sent.start: i for i, sent in enumerate(sentences)}

        for ent in doc.ents:
            # Find sentence ID
            sent_id = 0
            for start_pos, sid in sent_starts.items():
                if ent.start >= start_pos:
                    sent_id = sid

            mention = EntityMention(
                text=ent.text,
                label=ent.label_,
                start_pos=ent.start,
                end_pos=ent.end,
                sentence_id=sent_id,
            )
            mentions.append(mention)

        return mentions

    def _build_coreference_chains(
        self,
        doc: Doc,
        entity_mentions: list[EntityMention]
    ) -> list[CoreferenceChain]:
        """Build coreference chains using simple heuristics."""
        chains = []

        # Group mentions by text (simple string matching)
        mention_groups = defaultdict(list)
        for mention in entity_mentions:
            # Normalize text
            normalized = mention.text.lower().strip()
            mention_groups[normalized].append(mention)

        # Create chains for repeated entities
        chain_id = 0
        for text, mentions in mention_groups.items():
            if len(mentions) > 1:
                # Determine chain type
                chain_type = "proper" if mentions[0].label in ["PERSON", "ORG", "GPE"] else "nominal"

                chain = CoreferenceChain(
                    chain_id=chain_id,
                    mentions=mentions,
                    chain_type=chain_type,
                )
                chains.append(chain)
                chain_id += 1

        # Simple pronoun resolution (very basic)
        pronouns = ["he", "she", "it", "they", "him", "her", "them", "his", "her", "their"]
        sentences = list(doc.sents)

        for sent_id, sent in enumerate(sentences):
            for token in sent:
                if token.text.lower() in pronouns and token.pos_ == "PRON":
                    # Look for nearest entity in previous sentence
                    if sent_id > 0:
                        prev_mentions = [m for m in entity_mentions if m.sentence_id == sent_id - 1]
                        if prev_mentions:
                            # Create pronoun chain with most recent entity
                            pronoun_mention = EntityMention(
                                text=token.text,
                                label="PRONOUN",
                                start_pos=token.i,
                                end_pos=token.i + 1,
                                sentence_id=sent_id,
                            )

                            chain = CoreferenceChain(
                                chain_id=chain_id,
                                mentions=[prev_mentions[-1], pronoun_mention],
                                chain_type="pronoun",
                            )
                            chains.append(chain)
                            chain_id += 1

        return chains

    def _detect_discourse_relations(self, doc: Doc) -> list[DiscourseRelation]:
        """Detect discourse relations using connectives."""
        relations = []
        sentences = list(doc.sents)

        for i in range(len(sentences) - 1):
            sent1 = sentences[i]
            sent2 = sentences[i + 1]

            # Check for discourse connectives at start of sent2
            first_tokens = sent2.text.lower().split()[:3]

            relation_type = None
            confidence = 0.8

            for token in first_tokens:
                if token in self.CAUSAL_CONNECTIVES:
                    relation_type = "cause"
                    break
                elif token in self.CONTRAST_CONNECTIVES:
                    relation_type = "contrast"
                    break
                elif token in self.ELABORATION_CONNECTIVES:
                    relation_type = "elaboration"
                    break
                elif token in self.TEMPORAL_CONNECTIVES:
                    relation_type = "temporal"
                    break

            if relation_type:
                relation = DiscourseRelation(
                    relation_type=relation_type,
                    segment1=sent1.text,
                    segment2=sent2.text,
                    confidence=confidence,
                )
                relations.append(relation)

        return relations

    def _analyze_information_flow(
        self,
        doc: Doc,
        entity_mentions: list[EntityMention]
    ) -> list[InformationFlow]:
        """Analyze information flow (given vs. new)."""
        flow_list = []
        sentences = list(doc.sents)

        # Track entities seen so far
        seen_entities = set()

        for sent_id, sent in enumerate(sentences):
            # Get entities in this sentence
            sent_mentions = [m for m in entity_mentions if m.sentence_id == sent_id]
            sent_entities = {m.text.lower() for m in sent_mentions}

            # Classify as new or given
            new_entities = list(sent_entities - seen_entities)
            given_entities = list(sent_entities & seen_entities)

            # Simple topic detection (first entity or subject)
            topic = None
            focus = None

            if sent_mentions:
                topic = sent_mentions[0].text

            # Focus is often the subject
            for token in sent:
                if token.dep_ == "nsubj":
                    focus = token.text
                    break

            flow = InformationFlow(
                sentence_id=sent_id,
                new_entities=new_entities,
                given_entities=given_entities,
                topic=topic,
                focus=focus,
            )
            flow_list.append(flow)

            # Update seen entities
            seen_entities.update(sent_entities)

        return flow_list

    def _compute_entity_density(
        self,
        entity_mentions: list[EntityMention],
        doc: Doc
    ) -> float:
        """Compute entities per sentence."""
        num_sentences = len(list(doc.sents))
        if num_sentences == 0:
            return 0.0
        return len(entity_mentions) / num_sentences

    def _compute_avg_chain_length(self, chains: list[CoreferenceChain]) -> float:
        """Compute average coreference chain length."""
        if not chains:
            return 0.0
        total_length = sum(len(chain.mentions) for chain in chains)
        return total_length / len(chains)

    def _compute_topic_continuity(self, information_flow: list[InformationFlow]) -> float:
        """Compute topic continuity score."""
        if len(information_flow) < 2:
            return 0.0

        # Count topic shifts
        topics = [flow.topic for flow in information_flow if flow.topic]
        if len(topics) < 2:
            return 0.0

        # Topic continuity = 1 - (shifts / possible shifts)
        shifts = sum(1 for i in range(len(topics) - 1) if topics[i] != topics[i + 1])
        possible_shifts = len(topics) - 1

        return 1.0 - (shifts / possible_shifts) if possible_shifts > 0 else 0.0

    def _compute_coherence_score(
        self,
        entity_mentions: list[EntityMention],
        coreference_chains: list[CoreferenceChain],
        information_flow: list[InformationFlow]
    ) -> float:
        """
        Compute overall discourse coherence score.

        Based on:
        - Entity density
        - Coreference chain presence
        - Given/new information balance
        - Topic continuity
        """
        score = 0.0

        # Entity density (normalized)
        if information_flow:
            entity_density = len(entity_mentions) / len(information_flow)
            score += min(entity_density / 3.0, 0.25)  # Cap at 0.25

        # Coreference chains
        if coreference_chains:
            score += min(len(coreference_chains) / 5.0, 0.25)  # Cap at 0.25

        # Given/new balance
        if information_flow:
            given_ratio = sum(len(f.given_entities) for f in information_flow) / max(
                sum(len(f.new_entities) + len(f.given_entities) for f in information_flow), 1
            )
            # Ideal ratio is around 0.5-0.7
            if 0.4 <= given_ratio <= 0.8:
                score += 0.25
            else:
                score += 0.1

        # Topic continuity
        topic_continuity = self._compute_topic_continuity(information_flow)
        score += topic_continuity * 0.25

        return min(score, 1.0)  # Cap at 1.0

