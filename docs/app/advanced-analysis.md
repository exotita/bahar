# Advanced Linguistic Analysis

## Overview

The **Advanced Analysis** feature provides comprehensive linguistic analysis using four specialized analyzers. This goes beyond basic emotion detection to explore deep linguistic structures, semantics, morphology, and discourse patterns.

## What is Advanced Analysis?

Advanced Analysis combines multiple linguistic analysis techniques to provide insights into:

- **Lexical & Compositional Semantics**: Understanding word meanings and relationships
- **Morphology & Phonology**: Analyzing word structure and sound patterns
- **Distributional Semantics & Embeddings**: Exploring semantic spaces and word relationships
- **Pragmatics & Discourse**: Examining how language is used in context

## Key Features

### 📚 Lexical & Compositional Semantics

**What it analyzes:**
- **Word Sense Disambiguation**: Determining the correct meaning of ambiguous words
- **Semantic Similarity**: Measuring how similar words are in meaning
- **Lexical Chains**: Identifying sequences of semantically related words
- **Polysemy**: Detecting words with multiple meanings

**Metrics provided:**
- Lexical Diversity (0-1): Variety of vocabulary used
- Semantic Density (0-1): Concentration of meaningful content
- Polysemy Rate (0-1): Proportion of words with multiple meanings
- Cohesion Score (0-1): How well the text holds together semantically

**Example insights:**
```
Text: "The bank by the river has steep slopes."
- Word "bank" disambiguated as "riverbank" (not financial institution)
- Lexical chain: bank → river → slopes (geographical features)
- High semantic cohesion due to consistent theme
```

### 🔤 Morphology & Phonology

**What it analyzes:**
- **Morpheme Segmentation**: Breaking words into meaningful units
- **Affix Detection**: Identifying prefixes, suffixes, and infixes
- **Syllabification**: Dividing words into syllables
- **Phonological Features**: Analyzing sound patterns (consonants, vowels)

**Metrics provided:**
- Morphemes per Word: Average morphological complexity
- Morphological Complexity (0-1): Overall structural complexity
- Syllables per Word: Average syllable count
- Consonant/Vowel Ratio: Phonological balance

**Example insights:**
```
Text: "The researchers are investigating unprecedented phenomena."
- "researchers" → research + er + s (3 morphemes)
- "investigating" → in + vestig + ate + ing (4 morphemes)
- "unprecedented" → un + pre + cedent + ed (4 morphemes)
- High morphological complexity (many affixes)
```

### 🧠 Distributional Semantics & Embeddings

**What it analyzes:**
- **Word Embeddings**: 300-dimensional vector representations
- **Semantic Neighbors**: Words with similar meanings
- **Semantic Clustering**: Grouping related concepts
- **Dimensionality Analysis**: Understanding semantic space structure

**Metrics provided:**
- Vector Dimensionality: Number of dimensions (typically 300)
- Semantic Density (0-1): How tightly concepts are related
- Cluster Quality (0-1): How well-defined semantic groups are
- Effective Dimensions: Number of meaningful dimensions

**Example insights:**
```
Text: "Dogs and cats are popular pets. Many families adopt animals."
- Semantic cluster 1: dogs, cats, pets, animals (animals)
- Semantic cluster 2: popular, families, adopt (social)
- "dogs" neighbors: cats (0.85), puppies (0.78), pets (0.72)
- High semantic density (focused topic)
```

### 💬 Pragmatics & Discourse

**What it analyzes:**
- **Entity Tracking**: Following mentions of people, places, things
- **Coreference Resolution**: Linking pronouns to their referents
- **Discourse Relations**: Understanding how sentences connect
- **Information Flow**: Tracking given vs. new information

**Metrics provided:**
- Entity Density (0-1): Concentration of named entities
- Average Chain Length: How long entity references persist
- Topic Continuity (0-1): How consistently topics are maintained
- Coherence Score (0-1): Overall text coherence

**Example insights:**
```
Text: "John loves his dog. He walks it every morning."
- Entity: "John" (PERSON)
- Coreference chain: John → He (2 mentions)
- Coreference chain: dog → it (2 mentions)
- High topic continuity (consistent subject)
```

## How to Use

### Step 1: Enable Analyzers

Choose which analyzers to run by checking the boxes:

- ✅ **Lexical & Compositional Semantics** - Word meanings and relationships
- ✅ **Morphology & Phonology** - Word structure and sounds
- ✅ **Distributional Semantics & Embeddings** - Semantic spaces
- ✅ **Pragmatics & Discourse** - Context and coherence

**Tip**: Enable all analyzers for comprehensive analysis, or select specific ones for focused insights.

### Step 2: Configure Parameters

**Language**: Select English or Dutch
- English: Uses `en_core_web_lg` spaCy model
- Dutch: Uses `nl_core_news_lg` spaCy model

**Semantic Neighbors (k)**: Number of similar words to find (3-10)
- Lower values: Only closest neighbors
- Higher values: Broader semantic context

**Semantic Clusters**: Number of concept groups (2-5)
- Fewer clusters: Broader categories
- More clusters: Finer distinctions

### Step 3: Enter Text

Paste or type your text in the input area. Works best with:
- **Minimum**: 2-3 sentences (20+ words)
- **Optimal**: 1-2 paragraphs (50-200 words)
- **Maximum**: Several paragraphs (500+ words)

**Note**: Longer texts provide richer analysis but take more time.

### Step 4: Analyze

Click **"🔬 Perform Advanced Analysis"** and wait for results.

**Performance**:
- First analysis: ~10 seconds (loading models)
- Subsequent analyses: ~0.5 seconds (cached models)

### Step 5: Explore Results

Results are organized into sections:

1. **Summary Metrics**: Quick overview (text length, word count, analyzers used)
2. **Semantic Analysis**: Word senses, lexical chains, semantic metrics
3. **Morphology Analysis**: Morpheme breakdowns, syllable counts, phonological features
4. **Embedding Analysis**: Semantic neighbors, clusters, vector statistics
5. **Discourse Analysis**: Entity mentions, coreference chains, information flow

**Tip**: Click expanders (▼) to see detailed information for each section.

### Step 6: Export Results

Download your analysis in two formats:

**Academic Format (JSON)**:
- Flattened structure for statistical analysis
- All metrics as top-level fields
- Ready for CSV conversion
- Ideal for research and data analysis

**Complete Format (JSON)**:
- Full hierarchical structure
- All detailed information preserved
- Nested objects and arrays
- Ideal for programmatic processing

## Use Cases

### Academic Research

**Linguistics Studies**:
- Analyze morphological complexity across languages
- Study semantic relationships in specialized domains
- Investigate discourse patterns in different genres

**Computational Linguistics**:
- Evaluate text generation quality
- Compare human vs. machine-generated text
- Benchmark semantic coherence

### Content Analysis

**Writing Quality**:
- Assess lexical diversity and vocabulary richness
- Evaluate semantic coherence and flow
- Identify morphological complexity levels

**Text Comparison**:
- Compare semantic density across documents
- Analyze discourse structure differences
- Evaluate cohesion and coherence

### Language Learning

**Proficiency Assessment**:
- Measure morphological complexity
- Evaluate lexical diversity
- Assess discourse coherence

**Text Difficulty**:
- Determine appropriate reading levels
- Identify complex linguistic structures
- Suggest simplification opportunities

## Understanding the Metrics

### Semantic Metrics

**Lexical Diversity (0-1)**:
- 0.0-0.3: Low (repetitive vocabulary)
- 0.3-0.6: Medium (balanced vocabulary)
- 0.6-1.0: High (rich, varied vocabulary)

**Semantic Density (0-1)**:
- 0.0-0.3: Low (sparse content)
- 0.3-0.6: Medium (balanced content)
- 0.6-1.0: High (information-rich)

**Polysemy Rate (0-1)**:
- 0.0-0.3: Low (simple words)
- 0.3-0.6: Medium (mixed vocabulary)
- 0.6-1.0: High (complex, ambiguous words)

**Cohesion Score (0-1)**:
- 0.0-0.3: Low (disconnected ideas)
- 0.3-0.6: Medium (some connections)
- 0.6-1.0: High (well-connected text)

### Morphology Metrics

**Morphemes per Word**:
- 1.0-1.5: Simple words (mostly root words)
- 1.5-2.5: Moderate complexity (some affixes)
- 2.5+: High complexity (many affixes)

**Morphological Complexity (0-1)**:
- 0.0-0.3: Simple structure
- 0.3-0.6: Moderate structure
- 0.6-1.0: Complex structure

**Syllables per Word**:
- 1.0-1.5: Short words
- 1.5-2.5: Medium words
- 2.5+: Long words

**C/V Ratio**:
- <1.5: Vowel-heavy (easier to pronounce)
- 1.5-2.5: Balanced
- >2.5: Consonant-heavy (harder to pronounce)

### Embedding Metrics

**Semantic Density (0-1)**:
- 0.0-0.3: Loosely related concepts
- 0.3-0.6: Moderately related concepts
- 0.6-1.0: Tightly related concepts

**Cluster Quality (0-1)**:
- 0.0-0.3: Poorly defined groups
- 0.3-0.6: Moderately defined groups
- 0.6-1.0: Well-defined groups

### Discourse Metrics

**Entity Density (0-1)**:
- 0.0-0.1: Few entities (abstract text)
- 0.1-0.3: Moderate entities (balanced)
- 0.3+: Many entities (concrete text)

**Topic Continuity (0-1)**:
- 0.0-0.3: Frequent topic changes
- 0.3-0.6: Moderate continuity
- 0.6-1.0: Consistent topic

**Coherence Score (0-1)**:
- 0.0-0.3: Incoherent text
- 0.3-0.6: Moderately coherent
- 0.6-1.0: Highly coherent

## Tips & Best Practices

### For Best Results

1. **Text Length**: Use 50-200 words for optimal balance
2. **Language**: Ensure text matches selected language
3. **Quality**: Use well-formed sentences with proper grammar
4. **Topic**: Focused topics produce clearer patterns

### Performance Optimization

1. **Enable Only Needed Analyzers**: Faster analysis
2. **Use Caching**: Repeated analyses with same config are instant
3. **Adjust Parameters**: Lower k and clusters for faster results

### Interpreting Results

1. **Compare Metrics**: Look at multiple metrics together
2. **Consider Context**: Metrics depend on text type and purpose
3. **Use Expanders**: Explore detailed information for insights
4. **Export Data**: Download for deeper statistical analysis

## Limitations

### Current Limitations

- **Languages**: Only English and Dutch supported
- **Text Length**: Very long texts (>1000 words) may be slow
- **Domain**: General-purpose models may miss domain-specific terms
- **Coreference**: Basic resolution, not perfect for complex texts

### Known Issues

- Persian language not yet supported for advanced analysis
- Some technical terms may not have embeddings
- Morphological analysis works best for English

## Technical Details

### Models Used

**spaCy Models**:
- English: `en_core_web_lg` (300-dim vectors)
- Dutch: `nl_core_news_lg` (300-dim vectors)

**NLTK Resources**:
- WordNet for word sense disambiguation
- Brown corpus for semantic similarity
- Punkt tokenizer for sentence segmentation

**Algorithms**:
- K-means clustering for semantic grouping
- PCA for dimensionality analysis
- Cosine similarity for semantic neighbors

### Performance

**First Analysis** (cold start):
- Model loading: ~8 seconds
- Analysis: ~2 seconds
- Total: ~10 seconds

**Subsequent Analyses** (cached):
- Model loading: 0 seconds (cached)
- Analysis: ~0.5 seconds
- Total: ~0.5 seconds

**Memory Usage**:
- Per configuration: ~500MB-1GB
- Typical usage: ~2-4GB (2-4 configs)

## Support

For issues or questions:
- Check the [Streamlit Advanced Analysis Guide](../../guides/streamlit-advanced-analysis.md)
- Review the [Phase 1 Foundation Complete](../../guides/phase1-foundation-complete.md)
- Consult the [Advanced Linguistic Analysis Plan](../../guides/advanced-linguistic-analysis-plan.md)

---

**Last Updated**: 2025-01-XX
**Version**: 0.2.0
**Feature**: Advanced Linguistic Analysis

