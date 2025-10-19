#!/usr/bin/env python3
"""
Enhanced Emotion and Linguistic Analysis Demo.

Demonstrates comprehensive text analysis combining:
- GoEmotions sentiment analysis (28 emotions)
- Linguistic dimensions (formality, tone, intensity, style)

Suitable for academic linguistic research.
"""

from __future__ import annotations

from enhanced_classifier import (
    EnhancedEmotionClassifier,
    export_to_academic_format,
    format_enhanced_output,
)

# Sample texts demonstrating different linguistic dimensions
DEMO_TEXTS: list[dict[str, str]] = [
    {
        "text": "I hereby formally request your assistance with this matter. Your prompt attention would be greatly appreciated.",
        "description": "Formal, polite, professional",
    },
    {
        "text": "Hey! Thanks so much for helping me out, you're awesome! Really appreciate it!",
        "description": "Colloquial, friendly, enthusiastic",
    },
    {
        "text": "This is absolutely unacceptable! I demand an explanation immediately!",
        "description": "Formal but angry, high intensity, direct",
    },
    {
        "text": "I dunno, maybe we could try that if you want? Whatever works for you.",
        "description": "Colloquial, passive, low intensity",
    },
    {
        "text": "I'm terribly sorry to bother you, but if possible, could you perhaps help me?",
        "description": "Formal, apologetic, passive, kind",
    },
    {
        "text": "Shut up! I don't wanna hear it anymore, seriously!",
        "description": "Colloquial, rough, high intensity, direct",
    },
    {
        "text": "I believe this approach will yield optimal results. The data clearly supports this conclusion.",
        "description": "Formal, serious, assertive",
    },
    {
        "text": "Wow, I can't believe this happened! This is so exciting!!!",
        "description": "Colloquial, excited, high intensity",
    },
    {
        "text": "I'm kinda scared about what might happen. It's pretty worrying, tbh.",
        "description": "Colloquial, fearful, medium intensity",
    },
    {
        "text": "Your kindness and support mean the world to me. I'm deeply grateful for everything.",
        "description": "Formal, kind, grateful, medium intensity",
    },
]


def main() -> None:
    """Run enhanced analysis demo."""
    print("=" * 80)
    print("ENHANCED EMOTION & LINGUISTIC ANALYSIS DEMO")
    print("Combining GoEmotions + Academic Linguistic Dimensions")
    print("=" * 80)

    # Initialize classifier
    print("\nInitializing classifier...")
    print("Note: First run will download the model (~400MB)")

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

    # Analyze demo texts
    print("\n" + "=" * 80)
    print("ANALYZING SAMPLE TEXTS")
    print("=" * 80)

    for idx, sample in enumerate(DEMO_TEXTS, 1):
        print(f"\n[Example {idx}] {sample['description']}")
        result = classifier.analyze(sample["text"], top_k=3)
        print(format_enhanced_output(result))

        # Show academic export format for first example
        if idx == 1:
            print("\n" + "-" * 80)
            print("ACADEMIC EXPORT FORMAT (for research/CSV)")
            print("-" * 80)
            academic_data = export_to_academic_format(result)
            for key, value in academic_data.items():
                if isinstance(value, float):
                    print(f"  {key:30s}: {value:.4f}")
                else:
                    print(f"  {key:30s}: {value}")

    # Multilingual examples
    print("\n" + "=" * 80)
    print("MULTILINGUAL EXAMPLES")
    print("=" * 80)

    multilingual_samples = [
        {
            "lang": "English",
            "text": "I'm extremely grateful for your wonderful support!",
        },
        {
            "lang": "Dutch",
            "text": "Ik ben zo blij met dit geweldige nieuws!",
        },
        {
            "lang": "Persian",
            "text": "من از این خبر عالی خیلی خوشحالم!",
        },
    ]

    for sample in multilingual_samples:
        print(f"\n[{sample['lang']}]")
        result = classifier.analyze(sample["text"], top_k=3)
        print(format_enhanced_output(result))

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("\nFor custom text analysis, use:")
    print("  python classify_enhanced.py \"Your text here\"")
    print("=" * 80)


if __name__ == "__main__":
    main()

