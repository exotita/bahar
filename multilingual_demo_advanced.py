#!/usr/bin/env python3
"""
Bahar - Advanced Multilingual Emotion & Linguistic Analysis Demo

This script demonstrates all the new features:
- Multilingual support (English, Dutch, Persian)
- 9 language-specific models
- Enhanced linguistic analysis
- Model comparison
- Batch processing

Run this script to see all features in action!
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

from bahar import EmotionAnalyzer, EnhancedAnalyzer
from bahar.datasets.goemotions import GOEMOTIONS_EMOTIONS, EMOTION_GROUPS
from bahar.utils.language_models import (
    get_available_models,
    get_supported_languages,
    detect_language,
)
from bahar.analyzers.enhanced_analyzer import export_to_academic_format

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import track

console = Console()


def show_header():
    """Display welcome header."""
    console.print("\n" + "="*70)
    console.print("[bold cyan]🌸 Bahar - Advanced Multilingual Emotion Analysis Demo[/bold cyan]")
    console.print("="*70 + "\n")


def show_available_models():
    """Display all available models."""
    console.print("[bold cyan]🌍 Available Languages and Models[/bold cyan]\n")

    languages = get_supported_languages()

    for lang in languages:
        models = get_available_models(lang)

        table = Table(title=f"{lang.upper()} Models", show_header=True, header_style="bold magenta")
        table.add_column("Model Key", style="cyan", width=25)
        table.add_column("HuggingFace Model", style="white", width=50)

        for key, model_name in models.items():
            table.add_row(key, model_name)

        console.print(table)
        console.print()


def show_emotion_taxonomy():
    """Display GoEmotions taxonomy."""
    console.print("[bold cyan]🎭 GoEmotions Emotion Taxonomy[/bold cyan]\n")

    colors = {
        "positive": "green",
        "negative": "red",
        "ambiguous": "yellow",
        "neutral": "white"
    }

    for group, emotions in EMOTION_GROUPS.items():
        color = colors.get(group, "white")
        console.print(f"[bold {color}]{group.upper()}[/bold {color}] ({len(emotions)} emotions)")
        console.print(f"[{color}]{', '.join(emotions)}[/{color}]\n")


def demo_language_detection():
    """Demonstrate language detection."""
    console.print("[bold cyan]🔍 Language Detection Demo[/bold cyan]\n")

    test_texts = [
        "This is absolutely wonderful and amazing!",
        "Dit is absoluut verschrikkelijk en teleurstellend.",
        "این واقعاً افتضاح و ناامیدکننده است.",
    ]

    detection_table = Table(show_header=True, header_style="bold green")
    detection_table.add_column("Text", style="white", width=50)
    detection_table.add_column("Detected Language", style="cyan", width=20)

    for text in test_texts:
        detected = detect_language(text)
        detection_table.add_row(text, detected.upper())

    console.print(detection_table)
    console.print()


def demo_basic_analysis():
    """Demonstrate basic emotion analysis for each language."""
    console.print("[bold cyan]🎭 Basic Emotion Analysis[/bold cyan]\n")

    tests = [
        {
            "language": "english",
            "model_key": "goemotions",
            "text": "This is absolutely wonderful and amazing! I'm so excited!",
            "description": "English - GoEmotions"
        },
        {
            "language": "dutch",
            "model_key": "sentiment",
            "text": "Dit is absoluut verschrikkelijk en teleurstellend.",
            "description": "Dutch - Sentiment Model",
            "translation": "This is absolutely terrible and disappointing."
        },
        {
            "language": "persian",
            "model_key": "sentiment",
            "text": "این واقعاً افتضاح و ناامیدکننده است.",
            "description": "Persian - ParsBERT",
            "translation": "This is really terrible and disappointing."
        },
    ]

    for test in tests:
        console.print(f"[bold yellow]═══ {test['description']} ═══[/bold yellow]\n")
        console.print(f"[bold]Text:[/bold] {test['text']}")
        if "translation" in test:
            console.print(f"[dim]Translation: {test['translation']}[/dim]")
        console.print()

        analyzer = EmotionAnalyzer(language=test["language"], model_key=test["model_key"])
        analyzer.load_model()

        result = analyzer.analyze(test["text"], top_k=5)

        table = Table(show_header=True, header_style="bold green")
        table.add_column("Emotion", style="cyan", width=20)
        table.add_column("Score", style="yellow", width=10)
        table.add_column("Confidence Bar", style="white", width=30)

        for emotion, score in result.get_top_emotions():
            bar_length = int(score * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            table.add_row(emotion, f"{score:.4f}", bar)

        console.print(table)

        sentiment = result.get_sentiment_group()
        sentiment_colors = {"positive": "green", "negative": "red", "ambiguous": "yellow", "neutral": "white"}
        color = sentiment_colors.get(sentiment, "white")
        console.print(f"\n[bold]Sentiment:[/bold] [{color}]{sentiment.upper()}[/{color}]\n")


def demo_enhanced_analysis():
    """Demonstrate enhanced analysis with linguistics."""
    console.print("[bold cyan]🎨 Enhanced Analysis (Emotion + Linguistics)[/bold cyan]\n")

    test_cases = [
        {
            "text": "I am extremely disappointed with this terrible service.",
            "description": "Formal negative feedback"
        },
        {
            "text": "OMG this is sooo amazing!!! I love it!!!",
            "description": "Informal enthusiastic response"
        },
    ]

    analyzer = EnhancedAnalyzer(language="english", model_key="goemotions")
    analyzer.load_model()

    for idx, case in enumerate(test_cases, 1):
        console.print(f"[bold yellow]═══ Example {idx}: {case['description']} ═══[/bold yellow]\n")
        console.print(f"[bold]Text:[/bold] {case['text']}\n")

        result = analyzer.analyze(case['text'], top_k=3)

        # Emotion table
        emotion_table = Table(title="🎭 Emotions", show_header=True, header_style="bold green")
        emotion_table.add_column("Emotion", style="cyan", width=20)
        emotion_table.add_column("Score", style="yellow", width=10)

        for emotion, score in result.emotion_result.get_top_emotions():
            emotion_table.add_row(emotion, f"{score:.3f}")

        console.print(emotion_table)

        # Linguistic table
        ling_table = Table(title="📊 Linguistic Dimensions", show_header=True, header_style="bold magenta")
        ling_table.add_column("Dimension", style="cyan", width=20)
        ling_table.add_column("Value", style="yellow", width=15)

        ling = result.linguistic_features
        ling_table.add_row("Formality", ling.formality)
        ling_table.add_row("Tone", ling.tone)
        ling_table.add_row("Intensity", ling.intensity)
        ling_table.add_row("Style", ling.communication_style)

        console.print(ling_table)
        console.print()


def demo_multilingual_comparison():
    """Compare same sentiment across languages."""
    console.print("[bold cyan]🌐 Multilingual Comparison[/bold cyan]\n")
    console.print("[dim]Same sentiment expressed in different languages[/dim]\n")

    tests = [
        {
            "language": "english",
            "text": "I'm so happy and excited about this wonderful news!",
            "model_key": "goemotions"
        },
        {
            "language": "dutch",
            "text": "Ik ben zo blij en enthousiast over dit geweldige nieuws!",
            "model_key": "sentiment"
        },
        {
            "language": "persian",
            "text": "من از این خبر عالی خیلی خوشحالم و هیجان‌زده‌ام!",
            "model_key": "sentiment"
        },
    ]

    comparison_table = Table(show_header=True, header_style="bold green")
    comparison_table.add_column("Language", style="cyan", width=12)
    comparison_table.add_column("Text", style="white", width=40)
    comparison_table.add_column("Top Emotion", style="yellow", width=18)
    comparison_table.add_column("Sentiment", style="green", width=12)

    for test in track(tests, description="Analyzing..."):
        analyzer = EmotionAnalyzer(language=test["language"], model_key=test["model_key"])
        analyzer.load_model()

        result = analyzer.analyze(test["text"], top_k=1)
        top_emotion, top_score = result.get_top_emotions()[0]
        sentiment = result.get_sentiment_group()

        comparison_table.add_row(
            test["language"].upper(),
            test["text"][:40] + "...",
            f"{top_emotion} ({top_score:.2f})",
            sentiment.upper()
        )

    console.print(comparison_table)
    console.print()


def demo_batch_analysis():
    """Demonstrate batch processing."""
    console.print("[bold cyan]📈 Batch Analysis[/bold cyan]\n")

    batch_texts = [
        "This product exceeded all my expectations!",
        "I'm deeply disappointed with the quality.",
        "The service was okay, nothing special.",
        "I'm confused about how this works.",
        "Absolutely love it! Best purchase ever!",
    ]

    analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
    analyzer.load_model()

    results = analyzer.analyze_batch(batch_texts, top_k=2)

    batch_table = Table(show_header=True, header_style="bold green")
    batch_table.add_column("#", style="cyan", width=5)
    batch_table.add_column("Text", style="white", width=40)
    batch_table.add_column("Top Emotions", style="yellow", width=30)
    batch_table.add_column("Sentiment", style="magenta", width=12)

    for idx, (text, result) in enumerate(zip(batch_texts, results), 1):
        top_emotions = ", ".join([f"{e} ({s:.2f})" for e, s in result.get_top_emotions()])
        sentiment = result.get_sentiment_group()

        batch_table.add_row(
            str(idx),
            text[:40],
            top_emotions,
            sentiment.upper()
        )

    console.print(batch_table)
    console.print()


def demo_academic_export():
    """Demonstrate academic export format."""
    console.print("[bold cyan]💾 Academic Export Format[/bold cyan]\n")

    analyzer = EnhancedAnalyzer(language="english", model_key="goemotions")
    analyzer.load_model()

    text = "I'm extremely grateful for this amazing opportunity!"
    result = analyzer.analyze(text, top_k=5)

    academic_data = export_to_academic_format(result)

    export_table = Table(show_header=True, header_style="bold green")
    export_table.add_column("Field", style="cyan", width=30)
    export_table.add_column("Value", style="white", width=40)

    for key, value in list(academic_data.items())[:10]:
        if isinstance(value, float):
            export_table.add_row(key, f"{value:.4f}")
        else:
            export_table.add_row(key, str(value)[:40])

    console.print(export_table)
    console.print(f"\n[dim]Full export contains {len(academic_data)} fields[/dim]\n")


def main():
    """Run all demonstrations."""
    show_header()

    demos = [
        ("Available Models", show_available_models),
        ("Emotion Taxonomy", show_emotion_taxonomy),
        ("Language Detection", demo_language_detection),
        ("Basic Analysis", demo_basic_analysis),
        ("Enhanced Analysis", demo_enhanced_analysis),
        ("Multilingual Comparison", demo_multilingual_comparison),
        ("Batch Analysis", demo_batch_analysis),
        ("Academic Export", demo_academic_export),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            console.print(f"[red]✗ Error in {name}: {e}[/red]\n")

    # Summary
    console.print("="*70)
    console.print("[bold green]✓ Demo Complete![/bold green]")
    console.print("="*70)
    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  • Try the Streamlit app: [cyan]streamlit run app.py[/cyan]")
    console.print("  • Use CLI tools: [cyan]python classify_enhanced.py \"your text\"[/cyan]")
    console.print("  • Check Jupyter notebook: [cyan]multilingual_emotion_demo.ipynb[/cyan]")
    console.print()


if __name__ == "__main__":
    main()

