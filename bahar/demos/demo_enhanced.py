#!/usr/bin/env python3
"""
Enhanced Emotion and Linguistic Analysis Demo.

Demonstrates comprehensive text analysis combining:
- GoEmotions sentiment analysis (28 emotions)
- Linguistic dimensions (formality, tone, intensity, style)

Suitable for academic linguistic research.
"""

from __future__ import annotations

from rich.table import Table

from bahar.analyzers.enhanced_analyzer import (
    EnhancedAnalyzer,
    export_to_academic_format,
    format_enhanced_output,
)
from bahar.utils.rich_output import console, print_header, print_info, print_section, print_success

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
    print_header(
        "ENHANCED EMOTION & LINGUISTIC ANALYSIS DEMO",
        "Combining GoEmotions + Academic Linguistic Dimensions"
    )

    # Initialize classifier
    print_info("Initializing classifier...")
    print_info("Note: First run will download the model (~400MB)")

    classifier = EnhancedAnalyzer(emotion_dataset="goemotions")
    classifier.load_model()
    print_success("Model loaded successfully!")

    # Analyze demo texts
    console.print()
    print_section("ANALYZING SAMPLE TEXTS")

    for idx, sample in enumerate(DEMO_TEXTS, 1):
        console.print(f"\n[bold cyan][Example {idx}][/bold cyan] [dim]{sample['description']}[/dim]")
        result = classifier.analyze(sample["text"], top_k=3)
        format_enhanced_output(result, use_rich=True)

        # Show academic export format for first example
        if idx == 1:
            console.print()
            print_section("ACADEMIC EXPORT FORMAT (for research/CSV)")

            academic_data = export_to_academic_format(result)

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Field", style="yellow", width=35)
            table.add_column("Value", style="white", width=40)

            for key, value in academic_data.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))

            console.print(table)

    # Multilingual examples
    console.print()
    print_section("MULTILINGUAL EXAMPLES")

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
        console.print(f"\n[bold yellow][{sample['lang']}][/bold yellow]")
        result = classifier.analyze(sample["text"], top_k=3)
        format_enhanced_output(result, use_rich=True)

    console.print()
    print_success("Demo completed!")
    console.print("\n[bold]For custom text analysis, use:[/bold]")
    console.print("  [cyan]python classify_enhanced.py \"Your text here\"[/cyan]")


if __name__ == "__main__":
    main()

