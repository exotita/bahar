"""
Linguistic Analysis Module for Text Classification.

Extends emotion classification with linguistic dimensions:
- Formality (formal, colloquial, neutral)
- Tone (serious, friendly, rough, kind)
- Emotional intensity (high, medium, low)
- Communication style (direct, indirect, assertive, passive)

Designed for academic linguistic research and sentiment analysis.
"""

from __future__ import annotations

from typing import Final

# Linguistic dimension markers
FORMALITY_MARKERS: Final[dict[str, list[str]]] = {
    "formal": [
        "hereby",
        "therefore",
        "furthermore",
        "nevertheless",
        "consequently",
        "accordingly",
        "regarding",
        "pursuant",
        "notwithstanding",
        "shall",
        "ought",
        "whom",
        "thus",
        "henceforth",
    ],
    "colloquial": [
        "gonna",
        "wanna",
        "gotta",
        "yeah",
        "yep",
        "nope",
        "kinda",
        "sorta",
        "dunno",
        "ain't",
        "y'all",
        "lol",
        "omg",
        "btw",
        "tbh",
        "idk",
    ],
}

TONE_MARKERS: Final[dict[str, list[str]]] = {
    "friendly": [
        "thanks",
        "please",
        "appreciate",
        "wonderful",
        "lovely",
        "great",
        "awesome",
        "nice",
        "kind",
        "sweet",
        "dear",
        "friend",
    ],
    "rough": [
        "damn",
        "hell",
        "crap",
        "stupid",
        "idiot",
        "shut up",
        "get lost",
        "whatever",
        "seriously",
    ],
    "serious": [
        "important",
        "critical",
        "urgent",
        "significant",
        "essential",
        "crucial",
        "vital",
        "imperative",
        "must",
        "required",
    ],
    "kind": [
        "care",
        "help",
        "support",
        "understand",
        "comfort",
        "gentle",
        "tender",
        "compassion",
        "sympathy",
        "empathy",
    ],
}

# Emotion intensity markers
INTENSITY_MARKERS: Final[dict[str, list[str]]] = {
    "high": [
        "extremely",
        "absolutely",
        "completely",
        "totally",
        "incredibly",
        "amazingly",
        "terribly",
        "!!",
        "!!!",
        "very very",
        "so so",
    ],
    "medium": [
        "quite",
        "rather",
        "fairly",
        "pretty",
        "somewhat",
        "moderately",
        "reasonably",
    ],
    "low": [
        "slightly",
        "barely",
        "hardly",
        "scarcely",
        "a bit",
        "a little",
        "kind of",
        "sort of",
    ],
}


class LinguisticFeatures:
    """Container for linguistic analysis results."""

    def __init__(
        self,
        formality: str,
        formality_score: float,
        tone: str,
        tone_score: float,
        intensity: str,
        intensity_score: float,
        communication_style: str,
        style_score: float,
    ) -> None:
        self.formality: str = formality
        self.formality_score: float = formality_score
        self.tone: str = tone
        self.tone_score: float = tone_score
        self.intensity: str = intensity
        self.intensity_score: float = intensity_score
        self.communication_style: str = communication_style
        self.style_score: float = style_score

    def __repr__(self) -> str:
        return (
            f"LinguisticFeatures("
            f"formality={self.formality}({self.formality_score:.2f}), "
            f"tone={self.tone}({self.tone_score:.2f}), "
            f"intensity={self.intensity}({self.intensity_score:.2f}), "
            f"style={self.communication_style}({self.style_score:.2f}))"
        )


class LinguisticAnalyzer:
    """
    Analyzes linguistic dimensions of text.

    Provides academic-oriented linguistic analysis including:
    - Formality level
    - Tone characteristics
    - Emotional intensity
    - Communication style
    """

    def __init__(self) -> None:
        """Initialize the linguistic analyzer."""
        pass

    def analyze_formality(self, text: str) -> tuple[str, float]:
        """
        Analyze text formality level.

        Args:
            text: Input text

        Returns:
            Tuple of (formality_level, confidence_score)
        """
        text_lower = text.lower()

        formal_count = sum(
            1 for marker in FORMALITY_MARKERS["formal"] if marker in text_lower
        )
        colloquial_count = sum(
            1 for marker in FORMALITY_MARKERS["colloquial"] if marker in text_lower
        )

        # Check for additional formality indicators
        has_contractions = any(
            c in text for c in ["'t", "'re", "'ve", "'ll", "'d", "'m", "'s"]
        )
        if has_contractions:
            colloquial_count += 1

        # Check sentence structure complexity (simple heuristic)
        avg_word_length = sum(len(word) for word in text.split()) / max(
            len(text.split()), 1
        )
        if avg_word_length > 6:
            formal_count += 1
        elif avg_word_length < 4:
            colloquial_count += 1

        total = formal_count + colloquial_count
        if total == 0:
            return "neutral", 0.5

        formal_ratio = formal_count / total
        if formal_ratio > 0.6:
            return "formal", formal_ratio
        elif formal_ratio < 0.4:
            return "colloquial", 1 - formal_ratio
        else:
            return "neutral", 0.5

    def analyze_tone(self, text: str) -> tuple[str, float]:
        """
        Analyze text tone.

        Args:
            text: Input text

        Returns:
            Tuple of (tone_type, confidence_score)
        """
        text_lower = text.lower()

        tone_scores: dict[str, int] = {}
        for tone_type, markers in TONE_MARKERS.items():
            count = sum(1 for marker in markers if marker in text_lower)
            tone_scores[tone_type] = count

        # Check punctuation for tone indicators
        if "!" in text:
            tone_scores["friendly"] = tone_scores.get("friendly", 0) + 1
        if "?" in text and "please" in text_lower:
            tone_scores["friendly"] = tone_scores.get("friendly", 0) + 1

        total = sum(tone_scores.values())
        if total == 0:
            return "neutral", 0.5

        dominant_tone = max(tone_scores.items(), key=lambda x: x[1])
        confidence = dominant_tone[1] / total

        return dominant_tone[0], confidence

    def analyze_intensity(self, text: str) -> tuple[str, float]:
        """
        Analyze emotional intensity of text.

        Args:
            text: Input text

        Returns:
            Tuple of (intensity_level, confidence_score)
        """
        text_lower = text.lower()

        intensity_scores: dict[str, int] = {}
        for level, markers in INTENSITY_MARKERS.items():
            count = sum(1 for marker in markers if marker in text_lower)
            intensity_scores[level] = count

        # Check for capitalization (shouting)
        words = text.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(
            len(words), 1
        )
        if caps_ratio > 0.3:
            intensity_scores["high"] = intensity_scores.get("high", 0) + 2

        # Check for repeated punctuation
        if "!!" in text or "??" in text:
            intensity_scores["high"] = intensity_scores.get("high", 0) + 1

        total = sum(intensity_scores.values())
        if total == 0:
            return "medium", 0.5

        dominant_intensity = max(intensity_scores.items(), key=lambda x: x[1])
        confidence = dominant_intensity[1] / total

        return dominant_intensity[0], confidence

    def analyze_communication_style(self, text: str) -> tuple[str, float]:
        """
        Analyze communication style.

        Args:
            text: Input text

        Returns:
            Tuple of (style_type, confidence_score)
        """
        text_lower = text.lower()

        # Direct indicators
        direct_markers = ["must", "need to", "have to", "should", "will", "do"]
        direct_count = sum(1 for marker in direct_markers if marker in text_lower)

        # Indirect indicators
        indirect_markers = [
            "maybe",
            "perhaps",
            "might",
            "could",
            "possibly",
            "if you don't mind",
        ]
        indirect_count = sum(
            1 for marker in indirect_markers if marker in text_lower
        )

        # Assertive indicators
        assertive_markers = ["I think", "I believe", "in my opinion", "clearly"]
        assertive_count = sum(
            1 for marker in assertive_markers if marker in text_lower
        )

        # Passive indicators
        passive_markers = ["sorry", "excuse me", "if possible", "would you mind"]
        passive_count = sum(1 for marker in passive_markers if marker in text_lower)

        scores = {
            "direct": direct_count,
            "indirect": indirect_count,
            "assertive": assertive_count,
            "passive": passive_count,
        }

        total = sum(scores.values())
        if total == 0:
            return "neutral", 0.5

        dominant_style = max(scores.items(), key=lambda x: x[1])
        confidence = dominant_style[1] / total

        return dominant_style[0], confidence

    def analyze(self, text: str) -> LinguisticFeatures:
        """
        Perform complete linguistic analysis.

        Args:
            text: Input text

        Returns:
            LinguisticFeatures object with all analysis results
        """
        formality, formality_score = self.analyze_formality(text)
        tone, tone_score = self.analyze_tone(text)
        intensity, intensity_score = self.analyze_intensity(text)
        style, style_score = self.analyze_communication_style(text)

        return LinguisticFeatures(
            formality=formality,
            formality_score=formality_score,
            tone=tone,
            tone_score=tone_score,
            intensity=intensity,
            intensity_score=intensity_score,
            communication_style=style,
            style_score=style_score,
        )


def format_linguistic_analysis(features: LinguisticFeatures) -> str:
    """Format linguistic features for display."""
    lines: list[str] = []
    lines.append("\nLinguistic Analysis:")
    lines.append("-" * 60)

    # Formality
    bar = "█" * int(features.formality_score * 30)
    lines.append(
        f"  Formality:    {features.formality:12s} {bar:30s} {features.formality_score:.2f}"
    )

    # Tone
    bar = "█" * int(features.tone_score * 30)
    lines.append(
        f"  Tone:         {features.tone:12s} {bar:30s} {features.tone_score:.2f}"
    )

    # Intensity
    bar = "█" * int(features.intensity_score * 30)
    lines.append(
        f"  Intensity:    {features.intensity:12s} {bar:30s} {features.intensity_score:.2f}"
    )

    # Communication Style
    bar = "█" * int(features.style_score * 30)
    lines.append(
        f"  Style:        {features.communication_style:12s} {bar:30s} {features.style_score:.2f}"
    )

    return "\n".join(lines)

