# Streamlit Advanced Analysis Integration

## Overview

The **Advanced Analysis** tab in the Bahar Streamlit app provides an interactive interface for comprehensive linguistic analysis. This feature integrates all four advanced analyzers (Semantics, Morphology, Embeddings, Discourse) into a user-friendly web interface.

## Features

### 1. Interactive Configuration

#### Analyzer Selection
- **📚 Lexical & Compositional Semantics**: Word sense disambiguation, semantic similarity, lexical chains
- **🔤 Morphology & Phonology**: Morpheme segmentation, syllabification, phonological features
- **🧠 Distributional Semantics & Embeddings**: Word embeddings, semantic clustering, dimensionality analysis
- **💬 Pragmatics & Discourse**: Entity tracking, coreference, discourse relations, coherence

Each analyzer can be enabled or disabled independently via checkboxes.

#### Parameters
- **Language**: English or Dutch
- **Semantic Neighbors (k)**: Number of semantic neighbors to find (3-10)
- **Semantic Clusters**: Number of clusters for semantic space (2-5)

### 2. Analysis Results Display

#### Summary Metrics
- Text length (characters)
- Word count
- Number of analyzers used

#### Semantic Analysis
- **Metrics**: Lexical diversity, semantic density, polysemy rate, cohesion score
- **Details**:
  - Word senses with definitions, examples, and confidence scores
  - Lexical chains with type, words, strength, and length

#### Morphology Analysis
- **Metrics**: Morphemes per word, morphological complexity, syllables per word, consonant/vowel ratio
- **Details**:
  - Morpheme analysis showing word → lemma transformations
  - Morpheme breakdowns and affix identification

#### Embedding Analysis
- **Metrics**: Vector dimensions, semantic density, cluster quality, effective dimensions
- **Details**:
  - Semantic neighbors grouped by target word with similarity scores
  - Semantic clusters with cohesion scores and word lists

#### Discourse Analysis
- **Metrics**: Entity density, average chain length, topic continuity, coherence score
- **Details**:
  - Entity mentions table (entity, type, sentence)
  - Coreference chains showing mention sequences
  - Information flow analysis (topic, new entities, given entities)

### 3. Export Functionality

#### Academic Format (JSON)
Flattened structure optimized for statistical analysis and research:
- Text metadata (length, word count)
- All metrics as top-level fields
- Suitable for CSV conversion and data analysis

#### Complete Analysis (JSON)
Full hierarchical data structure:
- All analyzer results with complete details
- Nested structures preserved
- Suitable for programmatic processing

#### Preview
- Expandable section to preview academic format before downloading

### 4. Error Handling

- Clear error messages for analysis failures
- Expandable error details with full traceback
- Validation for empty text and disabled analyzers

## Usage

### Basic Workflow

1. **Navigate** to the "🔬 Advanced Analysis" tab
2. **Enable** desired analyzers using checkboxes
3. **Select** language and adjust parameters
4. **Enter** text for analysis
5. **Click** "🔬 Perform Advanced Analysis"
6. **Review** results in expandable sections
7. **Export** data in desired format

### Example Analysis

```python
# Sample text
text = """
Natural language processing enables computers to understand human language.
It combines linguistics, computer science, and artificial intelligence.
Modern NLP systems use deep learning and large language models.
"""

# Configuration
- Enable: All analyzers
- Language: English
- Semantic Neighbors: 5
- Semantic Clusters: 3

# Expected Results
- Semantic: High lexical diversity, technical vocabulary
- Morphology: Complex morphological structure, technical affixes
- Embeddings: Tight semantic clusters around "language", "processing", "systems"
- Discourse: Clear topic continuity, entity chains for "NLP", "computers"
```

## Technical Details

### Implementation

The Advanced Analysis tab is implemented in `app.py` as `tabs[1]`:

```python
with tabs[1]:
    # Configuration
    enable_semantics = st.checkbox("📚 Lexical & Compositional Semantics", ...)
    enable_morphology = st.checkbox("🔤 Morphology & Phonology", ...)
    enable_embeddings = st.checkbox("🧠 Distributional Semantics & Embeddings", ...)
    enable_discourse = st.checkbox("💬 Pragmatics & Discourse", ...)

    # Analysis
    analyzer = AdvancedLinguisticAnalyzer(
        language=lang_code,
        enable_semantics=enable_semantics,
        enable_morphology=enable_morphology,
        enable_embeddings=enable_embeddings,
        enable_discourse=enable_discourse,
    )
    analyzer.load_models()
    result = analyzer.analyze(text, top_k_neighbors=k, n_clusters=n)
```

### Dependencies

- `bahar.analyzers.advanced_analyzer.AdvancedLinguisticAnalyzer`
- `streamlit` for UI components
- `pandas` for data display
- `json` for export functionality

### Performance Considerations

- **Model Loading**: Models are loaded once per analyzer initialization and **cached automatically**
- **Caching**: Streamlit caches analyzers by configuration (language + enabled analyzers)
  - First analysis with a configuration: ~10 seconds
  - Subsequent analyses with same configuration: ~0.5 seconds (**95% faster**)
  - Different configuration: ~10 seconds (new cache entry)
- **Analysis Time**: Varies by text length and enabled analyzers
  - Semantics: ~1-2 seconds
  - Morphology: ~0.5-1 second
  - Embeddings: ~2-3 seconds
  - Discourse: ~1-2 seconds
- **Memory Usage**: ~500MB-1GB per cached configuration
- **Cache Management**: Clear cache via Streamlit menu (⋮ → Clear cache) or app restart

## Integration with Other Features

### Comparison with Basic Analysis

| Feature | Basic Analysis | Advanced Analysis |
|---------|---------------|-------------------|
| Emotion Detection | ✅ | ❌ |
| Linguistic Dimensions | ✅ | ❌ |
| NLP Features (spaCy) | ✅ | ❌ |
| Semantic Analysis | ❌ | ✅ |
| Morphology Analysis | ❌ | ✅ |
| Embeddings Analysis | ❌ | ✅ |
| Discourse Analysis | ❌ | ✅ |
| Export Format | JSON/CSV | JSON (2 formats) |

### Workflow Recommendation

1. **Start with Basic Analysis** for emotion and linguistic dimensions
2. **Use Advanced Analysis** for in-depth linguistic research
3. **Combine results** for comprehensive text understanding

## API Usage

For programmatic access without the UI:

```python
from bahar import AdvancedLinguisticAnalyzer

# Initialize
analyzer = AdvancedLinguisticAnalyzer(
    language="english",
    enable_semantics=True,
    enable_morphology=True,
    enable_embeddings=True,
    enable_discourse=True,
)

# Load models
analyzer.load_models()

# Analyze
result = analyzer.analyze(
    "Your text here",
    top_k_neighbors=5,
    n_clusters=3
)

# Get summary
summary = result.get_summary()

# Export
academic_data = result.export_academic_format()
complete_data = result.to_dict()
```

## Troubleshooting

### Common Issues

**Issue**: "No module named 'bahar.analyzers.advanced_analyzer'"
- **Solution**: Ensure Phase 1 implementation is complete and package is properly installed

**Issue**: "spaCy model not found"
- **Solution**: Download required models:
  ```bash
  python -m spacy download en_core_web_lg
  python -m spacy download nl_core_news_lg
  ```

**Issue**: "NLTK data not found"
- **Solution**: Download required NLTK data:
  ```python
  import nltk
  nltk.download('wordnet')
  nltk.download('omw-1.4')
  nltk.download('averaged_perceptron_tagger')
  ```

**Issue**: Analysis takes too long
- **Solution**:
  - Disable unused analyzers
  - Reduce text length
  - Lower `top_k_neighbors` and `n_clusters` parameters

**Issue**: Out of memory error
- **Solution**:
  - Analyze shorter texts
  - Enable fewer analyzers
  - Restart Streamlit app

## Future Enhancements

### Planned Features
- [ ] Visualization of semantic spaces (t-SNE/UMAP plots)
- [ ] Interactive word sense disambiguation
- [ ] Morpheme tree visualization
- [ ] Discourse graph visualization
- [ ] Batch processing for multiple texts
- [ ] CSV export format
- [ ] Comparison mode (analyze multiple texts side-by-side)
- [ ] Save/load analysis sessions
- [ ] Custom analyzer configurations

### Research Features
- [ ] Statistical significance testing
- [ ] Cross-linguistic comparison
- [ ] Corpus-level analysis
- [ ] Annotation export for linguistic research
- [ ] Integration with academic databases

## References

- [Advanced Linguistic Analysis Plan](./advanced-linguistic-analysis-plan.md)
- [Phase 1 Foundation Complete](./phase1-foundation-complete.md)
- [Dependency Installation Summary](./dependency-installation-summary.md)

## Support

For issues or questions:
- Check documentation in `docs/guides/`
- Review test script: `test_advanced_analysis.py`
- Consult API documentation in `docs/api/` (coming soon)

---

**Last Updated**: 2025-01-XX
**Version**: 0.2.0 (Phase 2)
**Status**: ✅ Complete and Ready for Testing

