#!/usr/bin/env python3
"""
Command-line utility for enhanced emotion and linguistic analysis.

Usage:
    python classify_enhanced.py "Your text here"
    python classify_enhanced.py "Your text" --top-k 5
    python classify_enhanced.py "Your text" --export-json
"""

from __future__ import annotations

import json
import sys

from enhanced_classifier import (
    EnhancedEmotionClassifier,
    export_to_academic_format,
    format_enhanced_output,
)


def main() -> None:
    """Parse arguments and perform enhanced analysis."""
    if len(sys.argv) < 2:
        print("Usage: python classify_enhanced.py <text> [--top-k N] [--export-json]")
        print("\nExamples:")
        print('  python classify_enhanced.py "I am so happy today!"')
        print('  python classify_enhanced.py "This is confusing" --top-k 5')
        print('  python classify_enhanced.py "Your text" --export-json')
        sys.exit(1)

    text = sys.argv[1]
    top_k = 3
    export_json_flag = False

    # Parse optional arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--top-k" and i + 1 < len(sys.argv):
            try:
                top_k = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print(f"Error: Invalid top-k value: {sys.argv[i + 1]}")
                sys.exit(1)
        elif sys.argv[i] == "--export-json":
            export_json_flag = True
            i += 1
        else:
            print(f"Error: Unknown argument: {sys.argv[i]}")
            sys.exit(1)

    # Initialize classifier
    print("Initializing enhanced classifier...")
    classifier = EnhancedEmotionClassifier()

    try:
        classifier.load_model()
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        print("\nTo install required dependencies, run:")
        print("  source .venv/bin/activate")
        print("  uv pip install transformers torch")
        sys.exit(1)

    # Perform analysis
    print("Analyzing text...\n")
    result = classifier.analyze(text, top_k=top_k)

    if export_json_flag:
        # Export as JSON
        academic_data = export_to_academic_format(result)
        print(json.dumps(academic_data, indent=2, ensure_ascii=False))
    else:
        # Display formatted output
        print(format_enhanced_output(result))


if __name__ == "__main__":
    main()

