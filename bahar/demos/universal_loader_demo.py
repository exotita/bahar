#!/usr/bin/env python3
"""
Universal Model Loader - Demo Script

Demonstrates how to use the universal model loader system to dynamically
load and use any HuggingFace model.
"""

from __future__ import annotations

from bahar.models import (
    ModelRegistry,
    UniversalModelLoader,
    ModelInspector,
    UniversalAdapter,
    ModelMetadata,
)
from bahar.utils.rich_output import console, print_header, print_section


def demo_add_model() -> None:
    """Demonstrate adding a new model to the registry."""
    print_header("DEMO 1", "Adding a New Model")

    # Initialize components
    registry = ModelRegistry()
    loader = UniversalModelLoader()

    # Model to add
    model_id = "cardiffnlp/twitter-roberta-base-emotion"

    console.print(f"\n[cyan]Loading model:[/cyan] {model_id}\n")

    # Load and inspect model
    model, tokenizer, config = loader.load_model(model_id)
    capabilities = ModelInspector.inspect_model(model, tokenizer, config)

    console.print(f"[green]✓ Model loaded successfully![/green]")
    console.print(f"  Task Type: {capabilities.task_type}")
    console.print(f"  Num Labels: {capabilities.num_labels}")
    console.print(f"  Architecture: {capabilities.architecture}\n")

    # Extract labels
    labels = ModelInspector.extract_labels(config)
    taxonomy = ModelInspector.detect_taxonomy(labels)

    console.print(f"[green]✓ Labels detected:[/green]")
    for idx, label in list(labels.items())[:5]:
        console.print(f"  {idx}: {label}")
    console.print(f"  ... ({len(labels)} total labels)")
    console.print(f"  Taxonomy: {taxonomy}\n")

    # Create metadata
    metadata = ModelMetadata(
        model_id=model_id,
        name="Twitter RoBERTa Emotion",
        description="RoBERTa model trained on Twitter data for emotion classification",
        task_type=capabilities.task_type,
        language=["english"],
        num_labels=capabilities.num_labels,
        label_map=labels,
        taxonomy=taxonomy,
        tags=["emotion", "twitter", "roberta", "english"],
    )

    # Add to registry
    try:
        registry.add_model(metadata)
        console.print(f"[green]✓ Model added to registry![/green]\n")
    except ValueError as e:
        console.print(f"[yellow]Model already in registry: {e}[/yellow]\n")


def demo_use_model() -> None:
    """Demonstrate using a model from the registry."""
    print_header("DEMO 2", "Using a Model from Registry")

    # Initialize components
    registry = ModelRegistry()
    loader = UniversalModelLoader()

    # Get model from registry
    model_id = "cardiffnlp/twitter-roberta-base-emotion"
    metadata = registry.get_model(model_id)

    if not metadata:
        console.print(f"[red]Model not found in registry. Run Demo 1 first.[/red]\n")
        return

    console.print(f"\n[cyan]Using model:[/cyan] {metadata.name}\n")

    # Load model
    model, tokenizer, config = loader.load_from_metadata(metadata)

    # Create adapter
    adapter = UniversalAdapter(model, tokenizer, metadata)

    # Test texts
    texts = [
        "I'm so happy and excited about this!",
        "This is terrible and disappointing.",
        "I'm not sure how I feel about this.",
    ]

    console.print("[cyan]Analyzing texts:[/cyan]\n")

    for text in texts:
        result = adapter.predict(text, top_k=3)

        console.print(f"[bold]Text:[/bold] {text}")
        console.print(f"[bold]Top predictions:[/bold]")
        for label, score in result.top_predictions:
            console.print(f"  • {label}: {score:.3f}")
        console.print()


def demo_list_models() -> None:
    """Demonstrate listing models in the registry."""
    print_header("DEMO 3", "Listing Models in Registry")

    registry = ModelRegistry()

    console.print(f"\n[cyan]Total models in registry:[/cyan] {len(registry)}\n")

    # List all models
    models = registry.list_models()

    if not models:
        console.print("[yellow]No models in registry yet.[/yellow]\n")
        return

    for metadata in models:
        console.print(f"[bold]{metadata.name}[/bold]")
        console.print(f"  ID: {metadata.model_id}")
        console.print(f"  Task: {metadata.task_type}")
        console.print(f"  Labels: {metadata.num_labels}")
        console.print(f"  Taxonomy: {metadata.taxonomy}")
        console.print(f"  Tags: {', '.join(metadata.tags)}")
        console.print()


def demo_batch_processing() -> None:
    """Demonstrate batch processing."""
    print_header("DEMO 4", "Batch Processing")

    registry = ModelRegistry()
    loader = UniversalModelLoader()

    model_id = "cardiffnlp/twitter-roberta-base-emotion"
    metadata = registry.get_model(model_id)

    if not metadata:
        console.print(f"[red]Model not found in registry. Run Demo 1 first.[/red]\n")
        return

    # Load model
    model, tokenizer, config = loader.load_from_metadata(metadata)
    adapter = UniversalAdapter(model, tokenizer, metadata)

    # Batch of texts
    texts = [
        "I love this!",
        "This is awful.",
        "Feeling neutral today.",
        "So excited!",
        "Very disappointed.",
    ]

    console.print(f"\n[cyan]Processing {len(texts)} texts in batch...[/cyan]\n")

    # Batch prediction
    results = adapter.predict_batch(texts, top_k=2)

    for i, (text, result) in enumerate(zip(texts, results), 1):
        console.print(f"[bold]{i}. {text}[/bold]")
        top_label, top_score = result.top_predictions[0]
        console.print(f"   → {top_label} ({top_score:.2%})\n")


def main() -> None:
    """Run all demos."""
    console.print("\n")
    console.print("=" * 80)
    console.print("[bold green]Universal Model Loader - Demo[/bold green]")
    console.print("=" * 80)
    console.print("\n")

    try:
        demo_add_model()
        demo_use_model()
        demo_list_models()
        demo_batch_processing()

        console.print("\n")
        console.print("=" * 80)
        console.print("[bold green]All demos completed successfully![/bold green]")
        console.print("=" * 80)
        console.print("\n")

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        raise


if __name__ == "__main__":
    main()

