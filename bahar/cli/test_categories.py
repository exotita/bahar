#!/usr/bin/env python3
"""
Test script for linguistic category recognition across multiple languages.

Tests all linguistic dimensions with multilingual samples.
"""

from __future__ import annotations

from rich.table import Table

from bahar.analyzers.enhanced_analyzer import (
    EnhancedAnalyzer,
    format_enhanced_output,
)
from bahar.analyzers.linguistic_samples import (
    LINGUISTIC_SAMPLES,
    get_all_categories,
    print_category_summary,
)
from bahar.utils.rich_output import (
    console,
    print_header,
    print_info,
    print_section,
    print_success,
)


def test_category_recognition(
    classifier: EnhancedAnalyzer,
    category: str,
    language: str = "english",
) -> None:
    """Test recognition for a specific category and language."""
    samples = LINGUISTIC_SAMPLES.get(category, [])

    if not samples:
        console.print(f"[yellow]No samples found for category: {category}[/yellow]")
        return

    print_section(f"CATEGORY: {category.upper()} ({language.upper()})")

    for idx, sample in enumerate(samples, 1):
        text = sample.get(language, "")
        if not text:
            continue

        console.print(f"\n[bold cyan][Sample {idx}][/bold cyan]")
        result = classifier.analyze(text, top_k=3)
        format_enhanced_output(result, use_rich=True)


def test_multilingual_comparison(
    classifier: EnhancedAnalyzer,
    category: str,
    sample_idx: int = 0,
) -> None:
    """Compare same text across different languages."""
    samples = LINGUISTIC_SAMPLES.get(category, [])

    if not samples or sample_idx >= len(samples):
        console.print(f"[yellow]Invalid category or sample index[/yellow]")
        return

    sample = samples[sample_idx]

    print_section(f"MULTILINGUAL COMPARISON: {category.upper()}")

    comparison_table = Table(show_header=True, header_style="bold cyan")
    comparison_table.add_column("Language", style="yellow", width=12)
    comparison_table.add_column("Emotion", style="green", width=15)
    comparison_table.add_column("Sentiment", style="magenta", width=12)
    comparison_table.add_column("Formality", style="cyan", width=12)
    comparison_table.add_column("Tone", style="blue", width=12)
    comparison_table.add_column("Intensity", style="yellow", width=12)

    for language in ["english", "dutch", "persian"]:
        text = sample.get(language, "")
        if not text:
            continue

        console.print(f"\n[bold cyan]{language.upper()}:[/bold cyan]")
        console.print(f"[italic]{text}[/italic]\n")

        result = classifier.analyze(text, top_k=3)

        # Add to comparison table
        top_emotion = result.emotion_result.get_top_emotions()[0][0]
        sentiment = result.emotion_result.get_sentiment_group()

        comparison_table.add_row(
            language.upper(),
            top_emotion,
            sentiment,
            result.linguistic_features.formality,
            result.linguistic_features.tone,
            result.linguistic_features.intensity
        )

    console.print(comparison_table)


def main() -> None:
    """Run comprehensive linguistic category tests."""
    print_header(
        "LINGUISTIC CATEGORY RECOGNITION TEST",
        "Testing emotion and linguistic analysis across multiple languages"
    )

    # Show category summary
    console.print()
    print_category_summary()

    # Initialize classifier
    console.print()
    print_info("Initializing enhanced classifier...")
    classifier = EnhancedAnalyzer(emotion_dataset="goemotions")

    try:
        classifier.load_model()
        print_success("Model loaded successfully!")
    except RuntimeError as exc:
        console.print(f"\n[red]Error: {exc}[/red]")
        console.print("\n[yellow]To install required dependencies, run:[/yellow]")
        console.print("  source .venv/bin/activate")
        console.print("  uv pip install transformers torch")
        return

    # Test each category in English
    console.print()
    print_header("TESTING ALL CATEGORIES IN ENGLISH", "Analyzing linguistic dimensions")

    categories_to_test = [
        "formal",
        "colloquial",
        "friendly",
        "rough",
        "serious",
        "kind",
        "high_intensity",
        "low_intensity",
        "direct",
        "passive",
        "sad",
        "scared",
        "surprised",
    ]

    for category in categories_to_test:
        test_category_recognition(classifier, category, "english")

    # Test multilingual comparison for key categories
    console.print()
    print_header("MULTILINGUAL COMPARISONS", "Comparing same texts across languages")

    comparison_categories = [
        ("formal", 0),
        ("friendly", 0),
        ("high_intensity", 0),
        ("sad", 0),
    ]

    for category, idx in comparison_categories:
        test_multilingual_comparison(classifier, category, idx)

    console.print()
    print_success("Testing completed!")


if __name__ == "__main__":
    main()

