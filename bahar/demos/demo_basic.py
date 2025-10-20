"""
Bahar - Multilingual Emotion Classification Demo

Demonstrates emotion classification using GoEmotions taxonomy
for Dutch, Persian, and English text.
"""

from __future__ import annotations

from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.datasets.goemotions.result import format_emotion_output
from bahar.datasets.goemotions.samples import SAMPLE_TEXTS
from bahar.utils.rich_output import print_header, print_info, print_section, print_success


def main() -> None:
    """Run emotion classification demo on sample texts."""
    print_header(
        "Bahar - Multilingual Emotion Classification",
        "Based on GoEmotions Dataset (28 emotions)"
    )

    # Initialize classifier
    print_info("Initializing emotion classifier...")
    print_info("Note: First run will download the model (~400MB)")

    classifier = EmotionAnalyzer(dataset="goemotions")
    classifier.load_model()
    print_success("Model loaded successfully!")

    # Process samples for each language
    for language in ["english", "dutch", "persian"]:
        print_section(f"{language.upper()} SAMPLES")

        samples = SAMPLE_TEXTS[language]
        for idx, sample in enumerate(samples, 1):
            print(f"\n[Sample {idx}]")
            result = classifier.analyze(sample["text"], top_k=3)
            format_emotion_output(result, use_rich=True)

            if "translation" in sample:
                print(f"\nTranslation: {sample['translation']}")
            print(f"Expected emotion: {sample['expected_emotion']}")

    print_success("\nDemo completed!")


if __name__ == "__main__":
    main()
