#!/usr/bin/env python3
"""
Test script for linguistic category recognition across multiple languages.

Tests all linguistic dimensions with multilingual samples.
"""

from __future__ import annotations

from enhanced_classifier import (
    EnhancedEmotionClassifier,
    format_enhanced_output,
)
from linguistic_samples import (
    LINGUISTIC_SAMPLES,
    get_all_categories,
    print_category_summary,
)


def test_category_recognition(
    classifier: EnhancedEmotionClassifier,
    category: str,
    language: str = "english",
) -> None:
    """Test recognition for a specific category and language."""
    samples = LINGUISTIC_SAMPLES.get(category, [])

    if not samples:
        print(f"No samples found for category: {category}")
        return

    print(f"\n{'=' * 80}")
    print(f"CATEGORY: {category.upper()} ({language.upper()})")
    print(f"{'=' * 80}")

    for idx, sample in enumerate(samples, 1):
        text = sample.get(language, "")
        if not text:
            continue

        print(f"\n[Sample {idx}]")
        result = classifier.analyze(text, top_k=3)
        print(format_enhanced_output(result))


def test_multilingual_comparison(
    classifier: EnhancedEmotionClassifier,
    category: str,
    sample_idx: int = 0,
) -> None:
    """Compare same text across different languages."""
    samples = LINGUISTIC_SAMPLES.get(category, [])

    if not samples or sample_idx >= len(samples):
        print(f"Invalid category or sample index")
        return

    sample = samples[sample_idx]

    print(f"\n{'=' * 80}")
    print(f"MULTILINGUAL COMPARISON: {category.upper()}")
    print(f"{'=' * 80}")

    for language in ["english", "dutch", "persian"]:
        text = sample.get(language, "")
        if not text:
            continue

        print(f"\n{language.upper()}:")
        print(f"Text: {text}")

        result = classifier.analyze(text, top_k=3)

        # Show key metrics
        print(f"\nEmotion: {result.emotion_result.get_top_emotions()[0][0]}")
        print(f"Sentiment: {result.emotion_result.get_sentiment_group()}")
        print(f"Formality: {result.linguistic_features.formality}")
        print(f"Tone: {result.linguistic_features.tone}")
        print(f"Intensity: {result.linguistic_features.intensity}")
        print(f"Style: {result.linguistic_features.communication_style}")
        print("-" * 80)


def main() -> None:
    """Run comprehensive linguistic category tests."""
    print("=" * 80)
    print("LINGUISTIC CATEGORY RECOGNITION TEST")
    print("Testing emotion and linguistic analysis across multiple languages")
    print("=" * 80)

    # Show category summary
    print("\n")
    print_category_summary()

    # Initialize classifier
    print("\nInitializing enhanced classifier...")
    classifier = EnhancedEmotionClassifier()

    try:
        classifier.load_model()
        print("Model loaded successfully!")
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        print("\nTo install required dependencies, run:")
        print("  source .venv/bin/activate")
        print("  uv pip install transformers torch")
        return

    # Test each category in English
    print("\n" + "=" * 80)
    print("TESTING ALL CATEGORIES IN ENGLISH")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print("MULTILINGUAL COMPARISONS")
    print("=" * 80)

    comparison_categories = [
        ("formal", 0),
        ("friendly", 0),
        ("high_intensity", 0),
        ("sad", 0),
    ]

    for category, idx in comparison_categories:
        test_multilingual_comparison(classifier, category, idx)

    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

