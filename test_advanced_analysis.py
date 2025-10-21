#!/usr/bin/env python3
"""
Quick test script for advanced linguistic analysis.

Tests all four analyzers with sample text.
"""

from bahar import AdvancedLinguisticAnalyzer
from bahar.utils.rich_output import console, print_header, print_section

def main() -> None:
    """Run quick test of advanced analysis."""

    print_header("🧪 TESTING ADVANCED LINGUISTIC ANALYSIS", "Phase 1 Verification")

    # Sample text
    text = """
    Natural language processing is fascinating. It combines linguistics, computer science,
    and artificial intelligence. Researchers study how computers can understand and generate
    human language. This field has many applications in translation, sentiment analysis,
    and information retrieval.
    """

    console.print()
    print_section("TEST TEXT")
    console.print(f"[dim]{text.strip()}[/dim]")
    console.print()

    # Initialize analyzer with all features
    print_section("INITIALIZING ANALYZER")
    console.print("Creating AdvancedLinguisticAnalyzer with all features enabled...")

    analyzer = AdvancedLinguisticAnalyzer(
        language="english",
        enable_semantics=True,
        enable_morphology=True,
        enable_embeddings=True,
        enable_discourse=True,
    )

    console.print(f"[green]✓ Analyzer created: {analyzer}[/green]")
    console.print()

    # Load models
    print_section("LOADING MODELS")
    console.print("Loading NLTK data, spaCy models, and other resources...")

    try:
        analyzer.load_models()
        console.print("[green]✓ All models loaded successfully![/green]")
    except Exception as e:
        console.print(f"[red]✗ Error loading models: {e}[/red]")
        return

    console.print()

    # Perform analysis
    print_section("PERFORMING ANALYSIS")
    console.print("Analyzing text with all four analyzers...")

    try:
        result = analyzer.analyze(text)
        console.print("[green]✓ Analysis complete![/green]")
    except Exception as e:
        console.print(f"[red]✗ Error during analysis: {e}[/red]")
        import traceback
        traceback.print_exc()
        return

    console.print()

    # Display summary
    print_section("ANALYSIS SUMMARY")

    summary = result.get_summary()

    console.print(f"[bold cyan]Text Statistics:[/bold cyan]")
    console.print(f"  • Length: {summary['text_length']} characters")
    console.print(f"  • Words: {summary['word_count']}")
    console.print()

    if 'semantic' in summary:
        console.print(f"[bold cyan]Semantic Analysis:[/bold cyan]")
        sem = summary['semantic']
        console.print(f"  • Lexical Diversity: {sem['lexical_diversity']:.3f}")
        console.print(f"  • Semantic Density: {sem['semantic_density']:.3f}")
        console.print(f"  • Polysemy Rate: {sem['polysemy_rate']:.3f}")
        console.print(f"  • Cohesion Score: {sem['cohesion_score']:.3f}")
        console.print(f"  • Disambiguated Words: {sem['disambiguated_words']}")
        console.print(f"  • Lexical Chains: {sem['lexical_chains']}")
        console.print()

    if 'morphology' in summary:
        console.print(f"[bold cyan]Morphology Analysis:[/bold cyan]")
        morph = summary['morphology']
        console.print(f"  • Morphemes per Word: {morph['morphemes_per_word']:.3f}")
        console.print(f"  • Morphological Complexity: {morph['morphological_complexity']:.3f}")
        console.print(f"  • Syllables per Word: {morph['syllables_per_word']:.3f}")
        console.print(f"  • C/V Ratio: {morph['consonant_vowel_ratio']:.3f}")
        console.print(f"  • Phonological Complexity: {morph['phonological_complexity']:.3f}")
        console.print()

    if 'embeddings' in summary:
        console.print(f"[bold cyan]Embedding Analysis:[/bold cyan]")
        emb = summary['embeddings']
        console.print(f"  • Vector Dimensionality: {emb['vector_dimensionality']}")
        console.print(f"  • Avg Vector Norm: {emb['avg_vector_norm']:.3f}")
        console.print(f"  • Semantic Density: {emb['semantic_density']:.3f}")
        console.print(f"  • Cluster Quality: {emb['cluster_quality']:.3f}")
        console.print(f"  • Effective Dimensions: {emb['effective_dimensions']}")
        console.print(f"  • Num Clusters: {emb['num_clusters']}")
        console.print()

    if 'discourse' in summary:
        console.print(f"[bold cyan]Discourse Analysis:[/bold cyan]")
        disc = summary['discourse']
        console.print(f"  • Entity Density: {disc['entity_density']:.3f}")
        console.print(f"  • Avg Chain Length: {disc['avg_chain_length']:.3f}")
        console.print(f"  • Topic Continuity: {disc['topic_continuity']:.3f}")
        console.print(f"  • Coherence Score: {disc['coherence_score']:.3f}")
        console.print(f"  • Num Entities: {disc['num_entities']}")
        console.print(f"  • Num Chains: {disc['num_chains']}")
        console.print()

    # Export test
    print_section("EXPORT TEST")

    console.print("[bold cyan]Academic Format Export:[/bold cyan]")
    academic_data = result.export_academic_format()
    console.print(f"  • Total metrics: {len(academic_data)}")
    console.print(f"  • Sample keys: {list(academic_data.keys())[:5]}")
    console.print()

    console.print("[green]✅ All tests passed successfully![/green]")
    console.print()

    print_section("NEXT STEPS")
    console.print("• Test with different languages (Dutch, Persian)")
    console.print("• Test individual analyzers")
    console.print("• Test batch processing")
    console.print("• Export to CSV/JSON files")
    console.print("• Integrate with Streamlit UI")
    console.print()


if __name__ == "__main__":
    main()

