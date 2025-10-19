#!/usr/bin/env python3
"""
Command-line utility for enhanced emotion and linguistic analysis (backward compatible wrapper).

For the actual implementation, see bahar/cli/classify_enhanced.py
"""

from __future__ import annotations

import json
import sys

from bahar.analyzers.enhanced_analyzer import (
    EnhancedAnalyzer,
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
    try:
        from bahar.utils.rich_output import print_info, print_success
        use_rich = True
        print_info("Initializing enhanced classifier...")
    except ImportError:
        use_rich = False
        print("Initializing enhanced classifier...")

    classifier = EnhancedAnalyzer(emotion_dataset="goemotions")

    try:
        classifier.load_model()
        if use_rich:
            print_success("Model loaded!")
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        print("\nTo install required dependencies, run:")
        print("  source .venv/bin/activate")
        print("  uv pip install transformers torch")
        sys.exit(1)

    # Perform analysis
    if use_rich:
        print_info("Analyzing text...")
        print()
    else:
        print("Analyzing text...\n")

    result = classifier.analyze(text, top_k=top_k)

    if export_json_flag:
        # Export as JSON
        academic_data = export_to_academic_format(result)
        print(json.dumps(academic_data, indent=2, ensure_ascii=False))
    else:
        # Display formatted output
        format_enhanced_output(result, use_rich=use_rich)


if __name__ == "__main__":
    main()
