#!/usr/bin/env python3
"""
Basic emotion classification demo (backward compatible wrapper).

For the actual implementation, see bahar/demos/demo_basic.py
"""

from __future__ import annotations

from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.datasets.goemotions.result import format_emotion_output
from bahar.datasets.goemotions.samples import SAMPLE_TEXTS


def main() -> None:
    """Run emotion classification demo on sample texts."""
    try:
        from bahar.utils.rich_output import (
            console,
            print_header,
            print_info,
            print_section,
            print_success,
        )
        use_rich = True
    except ImportError:
        use_rich = False

    if use_rich:
        print_header(
            "Bahar - Multilingual Emotion Classification",
            "Based on GoEmotions Dataset (28 emotions)"
        )
        print_info("Initializing emotion classifier...")
        print_info("Note: First run will download the model (~400MB)")
    else:
        print("=" * 80)
        print("Bahar - Multilingual Emotion Classification")
        print("Based on GoEmotions Dataset (27 emotions + neutral)")
        print("=" * 80)
        print("\nInitializing emotion classifier...")
        print("Note: First run will download the model (~400MB)")

    classifier = EmotionAnalyzer(dataset="goemotions")

    try:
        classifier.load_model()
        if use_rich:
            print_success("Model loaded successfully!")
        else:
            print("Model loaded successfully!")
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        print("\nTo install required dependencies, run:")
        print("  source .venv/bin/activate")
        print("  uv pip install transformers torch")
        return

    # Process samples for each language
    for language in ["english", "dutch", "persian"]:
        if use_rich:
            print_section(f"{language.upper()} SAMPLES")
        else:
            print("\n" + "=" * 80)
            print(f"{language.upper()} SAMPLES")
            print("=" * 80)

        samples = SAMPLE_TEXTS[language]
        for idx, sample in enumerate(samples, 1):
            if use_rich:
                console.print(f"\n[bold cyan][Sample {idx}][/bold cyan]")
            else:
                print(f"\n[Sample {idx}]")

            result = classifier.analyze(sample["text"], top_k=3)
            format_emotion_output(result, use_rich=use_rich)

            if "translation" in sample:
                if use_rich:
                    console.print(f"[dim]Translation: {sample['translation']}[/dim]")
                else:
                    print(f"\nTranslation: {sample['translation']}")

            if use_rich:
                console.print(f"[dim]Expected emotion: {sample['expected_emotion']}[/dim]")
            else:
                print(f"Expected emotion: {sample['expected_emotion']}")

    if use_rich:
        console.print()
        print_success("Demo completed!")
    else:
        print("\n" + "=" * 80)
        print("Demo completed!")
        print("=" * 80)


if __name__ == "__main__":
    main()
