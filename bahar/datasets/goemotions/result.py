"""Result classes for GoEmotions emotion classification."""

from __future__ import annotations

from bahar.datasets.goemotions.taxonomy import EMOTION_GROUPS
from bahar.utils.rich_output import (
    console,
    create_emotion_table,
    print_text_analysis,
)


class EmotionResult:
    """Result of emotion classification."""

    def __init__(
        self,
        text: str,
        emotions: dict[str, float],
        top_k: int = 3,
    ) -> None:
        self.text: str = text
        self.emotions: dict[str, float] = emotions
        self.top_k: int = top_k

    def get_top_emotions(self) -> list[tuple[str, float]]:
        """Get top-k emotions sorted by score."""
        sorted_emotions = sorted(
            self.emotions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_emotions[: self.top_k]

    def get_sentiment_group(self) -> str:
        """Determine overall sentiment group based on top emotion."""
        top_emotion = self.get_top_emotions()[0][0]
        for group, emotions in EMOTION_GROUPS.items():
            if top_emotion in emotions:
                return group
        return "neutral"

    def __repr__(self) -> str:
        top_emotions = self.get_top_emotions()
        emotions_str = ", ".join(
            [f"{emotion}: {score:.3f}" for emotion, score in top_emotions]
        )
        return f"EmotionResult(text='{self.text[:50]}...', top_emotions=[{emotions_str}], sentiment={self.get_sentiment_group()})"


def format_emotion_output(result: EmotionResult, use_rich: bool = True) -> str:
    """
    Format emotion result for display.

    Args:
        result: EmotionResult object
        use_rich: Use Rich formatting if True, plain text if False

    Returns:
        Formatted string (or prints directly if use_rich=True)
    """
    if use_rich:
        print_text_analysis(result.text)

        # Print sentiment
        sentiment = result.get_sentiment_group()
        sentiment_colors = {
            "positive": "green",
            "negative": "red",
            "ambiguous": "yellow",
            "neutral": "white"
        }
        console.print(
            f"[bold]Sentiment:[/bold] [{sentiment_colors.get(sentiment, 'white')}]{sentiment.upper()}[/{sentiment_colors.get(sentiment, 'white')}]"
        )

        # Print emotion table
        table = create_emotion_table(result)
        console.print(table)

        return ""  # Already printed

    # Plain text fallback
    lines: list[str] = []
    lines.append(f"\nText: {result.text}")
    lines.append(f"Sentiment Group: {result.get_sentiment_group().upper()}")
    lines.append("\nTop Emotions:")
    for emotion, score in result.get_top_emotions():
        bar_length = int(score * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        lines.append(f"  {emotion:15s} {bar} {score:.3f}")
    return "\n".join(lines)

