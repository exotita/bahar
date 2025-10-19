"""
Rich output formatting utilities.

Provides beautiful terminal output using the Rich library.
Reference: https://pypi.org/project/rich/
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Global console instance
console = Console()


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a formatted header."""
    if subtitle:
        text = Text()
        text.append(title, style="bold cyan")
        text.append("\n")
        text.append(subtitle, style="dim")
        console.print(Panel(text, border_style="cyan"))
    else:
        console.print(Panel(title, style="bold cyan", border_style="cyan"))


def print_section(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold yellow]{title}[/bold yellow]")
    console.print("─" * 80, style="dim")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def create_emotion_table(
    result,
    show_all: bool = False,
) -> Table:
    """
    Create a Rich table for emotion results.

    Args:
        result: EmotionResult object
        show_all: Show all emotions or just top-k

    Returns:
        Rich Table object
    """
    table = Table(title=f"Emotion Analysis", show_header=True, header_style="bold magenta")

    table.add_column("Emotion", style="cyan", width=15)
    table.add_column("Score", justify="right", style="green", width=10)
    table.add_column("Confidence", width=50)

    emotions_to_show = (
        sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)
        if show_all
        else result.get_top_emotions()
    )

    for emotion, score in emotions_to_show:
        # Create visual bar
        bar_length = int(score * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)

        # Color based on score
        if score > 0.7:
            bar_style = "bright_green"
        elif score > 0.4:
            bar_style = "yellow"
        else:
            bar_style = "dim"

        table.add_row(
            emotion,
            f"{score:.3f}",
            Text(bar, style=bar_style)
        )

    return table


def create_linguistic_table(features) -> Table:
    """
    Create a Rich table for linguistic features.

    Args:
        features: LinguisticFeatures object

    Returns:
        Rich Table object
    """
    table = Table(title="Linguistic Analysis", show_header=True, header_style="bold cyan")

    table.add_column("Dimension", style="cyan", width=20)
    table.add_column("Value", style="yellow", width=15)
    table.add_column("Confidence", width=45)

    dimensions = [
        ("Formality", features.formality, features.formality_score),
        ("Tone", features.tone, features.tone_score),
        ("Intensity", features.intensity, features.intensity_score),
        ("Style", features.communication_style, features.style_score),
    ]

    for dimension, value, score in dimensions:
        bar_length = int(score * 35)
        bar = "█" * bar_length + "░" * (35 - bar_length)

        # Color based on score
        if score > 0.7:
            bar_style = "bright_cyan"
        elif score > 0.5:
            bar_style = "cyan"
        else:
            bar_style = "dim"

        table.add_row(
            dimension,
            value,
            Text(bar, style=bar_style)
        )

    return table


def create_summary_panel(result) -> Panel:
    """
    Create a summary panel for enhanced results.

    Args:
        result: EnhancedAnalysisResult object

    Returns:
        Rich Panel object
    """
    summary = result.get_summary()

    text = Text()
    text.append("Primary Emotion: ", style="bold")
    text.append(summary['top_emotions'][0][0], style="bright_green")
    text.append("\n")

    text.append("Sentiment: ", style="bold")
    sentiment_colors = {
        "positive": "green",
        "negative": "red",
        "ambiguous": "yellow",
        "neutral": "white"
    }
    text.append(
        summary['sentiment_group'],
        style=sentiment_colors.get(summary['sentiment_group'], "white")
    )
    text.append("\n")

    text.append("Formality: ", style="bold")
    text.append(summary['formality'], style="cyan")
    text.append(" | ", style="dim")

    text.append("Tone: ", style="bold")
    text.append(summary['tone'], style="cyan")
    text.append("\n")

    text.append("Intensity: ", style="bold")
    text.append(summary['intensity'], style="magenta")
    text.append(" | ", style="dim")

    text.append("Style: ", style="bold")
    text.append(summary['communication_style'], style="magenta")

    return Panel(text, title="[bold]Summary[/bold]", border_style="green")


def print_text_analysis(text: str, max_length: int = 100) -> None:
    """Print the analyzed text in a formatted way."""
    if len(text) > max_length:
        display_text = text[:max_length] + "..."
    else:
        display_text = text

    console.print(Panel(
        display_text,
        title="[bold]Text[/bold]",
        border_style="blue",
        padding=(1, 2)
    ))


def create_progress_bar(description: str = "Processing") -> Progress:
    """
    Create a Rich progress bar.

    Args:
        description: Description text

    Returns:
        Rich Progress object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )


def print_model_loading() -> None:
    """Print model loading message."""
    with console.status("[bold green]Loading model...", spinner="dots"):
        import time
        time.sleep(0.5)  # Brief pause for visual effect
    print_success("Model loaded successfully!")


def print_category_summary(categories: dict[str, int]) -> None:
    """
    Print a summary of categories.

    Args:
        categories: Dictionary of category names and counts
    """
    table = Table(title="Category Summary", show_header=True, header_style="bold")

    table.add_column("Category", style="cyan", width=30)
    table.add_column("Count", justify="right", style="green", width=10)
    table.add_column("Distribution", width=40)

    total = sum(categories.values())

    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100 if total > 0 else 0
        bar_length = int(percentage / 2.5)  # Scale to fit
        bar = "█" * bar_length

        table.add_row(
            category,
            str(count),
            Text(f"{bar} {percentage:.1f}%", style="bright_blue")
        )

    console.print(table)
    console.print(f"\n[bold]Total:[/bold] {total} items")

