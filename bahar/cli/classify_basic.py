#!/usr/bin/env python3
"""
Command-line utility to classify custom text for emotions.

Usage:
    python classify_text.py "Your text here"
    python classify_text.py "Your text" --top-k 5
"""

from __future__ import annotations

import sys

from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.datasets.goemotions.result import format_emotion_output
from bahar.utils.rich_output import print_info, print_success


def classify_custom_text(text: str, top_k: int = 3) -> None:
    """Classify a custom text and display results."""
    print_info("Initializing emotion classifier...")

    classifier = EmotionAnalyzer(dataset="goemotions")

    classifier.load_model()
    print_success("Model loaded!")

    print_info("Classifying text...")
    print()

    result = classifier.analyze(text, top_k=top_k)
    format_emotion_output(result, use_rich=True)


def main() -> None:
    """Parse arguments and classify text."""
    if len(sys.argv) < 2:
        print("Usage: python classify_text.py <text> [--top-k N]")
        print("\nExample:")
        print('  python classify_text.py "I am so happy today!"')
        print('  python classify_text.py "This is confusing" --top-k 5')
        sys.exit(1)

    text = sys.argv[1]
    top_k = 3

    # Parse optional --top-k argument
    if len(sys.argv) >= 4 and sys.argv[2] == "--top-k":
        try:
            top_k = int(sys.argv[3])
        except ValueError:
            print(f"Error: Invalid top-k value: {sys.argv[3]}")
            sys.exit(1)

    classify_custom_text(text, top_k)


if __name__ == "__main__":
    main()

