"""
Enhanced analyzer combining emotion and linguistic analysis.

Provides comprehensive text analysis for academic research.
"""

from __future__ import annotations

from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.analyzers.linguistic_analyzer import (
    LinguisticAnalyzer,
    LinguisticFeatures,
)
from bahar.datasets.goemotions.result import EmotionResult
from bahar.utils.rich_output import (
    console,
    create_emotion_table,
    create_linguistic_table,
    create_summary_panel,
    print_header,
    print_section,
    print_text_analysis,
)


class EnhancedAnalysisResult:
    """Combined emotion and linguistic analysis result."""

    def __init__(
        self,
        text: str,
        emotion_result: EmotionResult,
        linguistic_features: LinguisticFeatures,
    ) -> None:
        self.text: str = text
        self.emotion_result: EmotionResult = emotion_result
        self.linguistic_features: LinguisticFeatures = linguistic_features

    def get_summary(self) -> dict[str, str | list[tuple[str, float]]]:
        """Get a summary of all analysis results."""
        return {
            "text": self.text,
            "sentiment_group": self.emotion_result.get_sentiment_group(),
            "top_emotions": self.emotion_result.get_top_emotions(),
            "formality": self.linguistic_features.formality,
            "tone": self.linguistic_features.tone,
            "intensity": self.linguistic_features.intensity,
            "communication_style": self.linguistic_features.communication_style,
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"EnhancedAnalysisResult(\n"
            f"  text='{self.text[:50]}...',\n"
            f"  sentiment={summary['sentiment_group']},\n"
            f"  top_emotion={summary['top_emotions'][0][0]},\n"
            f"  formality={summary['formality']},\n"
            f"  tone={summary['tone']},\n"
            f"  intensity={summary['intensity']},\n"
            f"  style={summary['communication_style']}\n"
            f")"
        )


class EnhancedAnalyzer:
    """
    Enhanced analyzer combining emotion detection and linguistic analysis.

    Provides:
    - Fine-grained emotion classification (28 emotions from GoEmotions)
    - Linguistic formality analysis
    - Tone detection
    - Emotional intensity measurement
    - Communication style identification

    Suitable for academic linguistic research and comprehensive sentiment analysis.
    """

    def __init__(
        self,
        language: str | None = None,
        model_key: str | None = None,
        model_name: str | None = None,
        auto_detect_language: bool = True,
    ) -> None:
        """
        Initialize the enhanced analyzer with multilingual support.

        Args:
            language: Language code ("english", "dutch", "persian"). Auto-detected if None.
            model_key: Model key for the language (e.g., "goemotions", "sentiment").
            model_name: Explicit HuggingFace model name (overrides language/model_key).
            auto_detect_language: Automatically detect language from input text.
        """
        self.emotion_analyzer = EmotionAnalyzer(
            language=language,
            model_key=model_key,
            model_name=model_name,
            auto_detect_language=auto_detect_language,
        )
        self.linguistic_analyzer = LinguisticAnalyzer()
        self._loaded: bool = False

    def load_model(self) -> None:
        """Load the emotion classification model."""
        if not self._loaded:
            self.emotion_analyzer.load_model()
            self._loaded = True

    def analyze(self, text: str, top_k: int = 3) -> EnhancedAnalysisResult:
        """
        Perform comprehensive analysis on text.

        Args:
            text: Input text in any supported language
            top_k: Number of top emotions to return

        Returns:
            EnhancedAnalysisResult with emotion and linguistic analysis
        """
        if not self._loaded:
            self.load_model()

        # Perform emotion classification
        emotion_result = self.emotion_analyzer.analyze(text, top_k=top_k)

        # Perform linguistic analysis
        linguistic_features = self.linguistic_analyzer.analyze(text)

        return EnhancedAnalysisResult(
            text=text,
            emotion_result=emotion_result,
            linguistic_features=linguistic_features,
        )

    def analyze_batch(
        self, texts: list[str], top_k: int = 3
    ) -> list[EnhancedAnalysisResult]:
        """
        Analyze multiple texts.

        Args:
            texts: List of input texts
            top_k: Number of top emotions to return

        Returns:
            List of EnhancedAnalysisResult objects
        """
        return [self.analyze(text, top_k=top_k) for text in texts]


def format_enhanced_output(result: EnhancedAnalysisResult, use_rich: bool = True) -> str:
    """
    Format enhanced analysis result for display.

    Args:
        result: EnhancedAnalysisResult object
        use_rich: Use Rich formatting if True, plain text if False

    Returns:
        Formatted string (or prints directly if use_rich=True)
    """
    if use_rich:
        # Header
        print_header(
            "COMPREHENSIVE TEXT ANALYSIS",
            "Emotion Classification + Linguistic Analysis"
        )

        # Text
        print_text_analysis(result.text)

        # Emotion Analysis
        print_section("EMOTION ANALYSIS (GoEmotions)")

        sentiment = result.emotion_result.get_sentiment_group()
        sentiment_colors = {
            "positive": "green",
            "negative": "red",
            "ambiguous": "yellow",
            "neutral": "white"
        }
        console.print(
            f"[bold]Sentiment:[/bold] [{sentiment_colors.get(sentiment, 'white')}]{sentiment.upper()}[/{sentiment_colors.get(sentiment, 'white')}]"
        )

        table = create_emotion_table(result.emotion_result)
        console.print(table)

        # Linguistic Analysis
        print_section("LINGUISTIC ANALYSIS (Academic Dimensions)")
        table = create_linguistic_table(result.linguistic_features)
        console.print(table)

        # Summary
        console.print()
        panel = create_summary_panel(result)
        console.print(panel)

        return ""  # Already printed

    # Plain text fallback
    lines: list[str] = []

    # Header
    lines.append("\n" + "=" * 80)
    lines.append("COMPREHENSIVE TEXT ANALYSIS")
    lines.append("=" * 80)

    # Text
    lines.append(f"\nText: {result.text}")

    # Emotion Analysis
    lines.append("\n" + "-" * 80)
    lines.append("EMOTION ANALYSIS (GoEmotions)")
    lines.append("-" * 80)
    lines.append(
        f"Sentiment Group: {result.emotion_result.get_sentiment_group().upper()}"
    )
    lines.append("\nTop Emotions:")
    for emotion, score in result.emotion_result.get_top_emotions():
        bar_length = int(score * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        lines.append(f"  {emotion:15s} {bar} {score:.3f}")

    # Linguistic Analysis
    lines.append("\n" + "-" * 80)
    lines.append("LINGUISTIC ANALYSIS (Academic Dimensions)")
    lines.append("-" * 80)

    features = result.linguistic_features

    # Formality
    bar_length = int(features.formality_score * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    lines.append(
        f"  Formality:    {features.formality:12s} {bar} {features.formality_score:.3f}"
    )

    # Tone
    bar_length = int(features.tone_score * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    lines.append(
        f"  Tone:         {features.tone:12s} {bar} {features.tone_score:.3f}"
    )

    # Intensity
    bar_length = int(features.intensity_score * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    lines.append(
        f"  Intensity:    {features.intensity:12s} {bar} {features.intensity_score:.3f}"
    )

    # Communication Style
    bar_length = int(features.style_score * 50)
    bar = "█" * bar_length + "░" * (50 - bar_length)
    lines.append(
        f"  Style:        {features.communication_style:12s} {bar} {features.style_score:.3f}"
    )

    # Summary
    lines.append("\n" + "-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    summary = result.get_summary()
    lines.append(f"  Primary Emotion:      {summary['top_emotions'][0][0]}")
    lines.append(f"  Sentiment:            {summary['sentiment_group']}")
    lines.append(f"  Formality Level:      {summary['formality']}")
    lines.append(f"  Tone:                 {summary['tone']}")
    lines.append(f"  Emotional Intensity:  {summary['intensity']}")
    lines.append(f"  Communication Style:  {summary['communication_style']}")

    lines.append("=" * 80)

    return "\n".join(lines)


def export_to_academic_format(
    result: EnhancedAnalysisResult,
) -> dict[str, str | float | list[tuple[str, float]]]:
    """
    Export analysis results in structured format for academic research.

    Returns:
        Dictionary with all analysis dimensions suitable for CSV/JSON export
    """
    summary = result.get_summary()
    top_emotions = summary["top_emotions"]

    return {
        "text": result.text,
        "text_length": len(result.text),
        "word_count": len(result.text.split()),
        # Emotion dimensions
        "sentiment_group": summary["sentiment_group"],
        "primary_emotion": top_emotions[0][0],
        "primary_emotion_score": top_emotions[0][1],
        "secondary_emotion": top_emotions[1][0] if len(top_emotions) > 1 else "",
        "secondary_emotion_score": top_emotions[1][1] if len(top_emotions) > 1 else 0.0,
        "tertiary_emotion": top_emotions[2][0] if len(top_emotions) > 2 else "",
        "tertiary_emotion_score": top_emotions[2][1] if len(top_emotions) > 2 else 0.0,
        # Linguistic dimensions
        "formality": result.linguistic_features.formality,
        "formality_score": result.linguistic_features.formality_score,
        "tone": result.linguistic_features.tone,
        "tone_score": result.linguistic_features.tone_score,
        "intensity": result.linguistic_features.intensity,
        "intensity_score": result.linguistic_features.intensity_score,
        "communication_style": result.linguistic_features.communication_style,
        "communication_style_score": result.linguistic_features.style_score,
    }

