"""
Bahar - Multilingual Emotion Classification Demo

Demonstrates emotion classification using GoEmotions taxonomy
for Dutch, Persian, and English text.
"""

from __future__ import annotations

from emotion_classifier import (
    MultilingualEmotionClassifier,
    format_emotion_output,
)
from sample_texts import SAMPLE_TEXTS


def main() -> None:
    """Run emotion classification demo on sample texts."""
    print("=" * 80)
    print("Bahar - Multilingual Emotion Classification")
    print("Based on GoEmotions Dataset (27 emotions + neutral)")
    print("=" * 80)

    # Initialize classifier
    print("\nInitializing emotion classifier...")
    print("Note: First run will download the model (~400MB)")

    classifier = MultilingualEmotionClassifier()

    try:
        classifier.load_model()
        print("Model loaded successfully!")
    except RuntimeError as exc:
        print(f"\nError: {exc}")
        print("\nTo install required dependencies, run:")
        print("  source .venv/bin/activate")
        print("  uv pip install transformers torch")
        return

    # Process samples for each language
    for language in ["english", "dutch", "persian"]:
        print("\n" + "=" * 80)
        print(f"{language.upper()} SAMPLES")
        print("=" * 80)

        samples = SAMPLE_TEXTS[language]
        for idx, sample in enumerate(samples, 1):
            print(f"\n[Sample {idx}]")
            result = classifier.predict(sample["text"], top_k=3)
            print(format_emotion_output(result))

            if "translation" in sample:
                print(f"\nTranslation: {sample['translation']}")
            print(f"Expected emotion: {sample['expected_emotion']}")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
