#!/usr/bin/env python3
"""
Command-line utility to classify custom text for emotions.

Usage:
    python classify_text.py "Your text here"
    python classify_text.py "Your text" --top-k 5
"""

from __future__ import annotations

import sys

from emotion_classifier import (
    MultilingualEmotionClassifier,
    format_emotion_output,
)


def classify_custom_text(text: str, top_k: int = 3) -> None:
    """Classify a custom text and display results."""
    print("Initializing emotion classifier...")

    classifier = MultilingualEmotionClassifier()

    try:
        classifier.load_model()
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        print("\nTo install required dependencies, run:")
        print("  source .venv/bin/activate")
        print("  uv pip install transformers torch")
        sys.exit(1)

    print("Classifying text...\n")
    result = classifier.predict(text, top_k=top_k)
    print(format_emotion_output(result))


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

