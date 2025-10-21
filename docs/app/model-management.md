# Model Management

## Overview

The **Model Management** feature allows you to dynamically add, configure, and test machine learning models from HuggingFace Hub. This provides flexibility to use any emotion or sentiment analysis model without code changes.

## What is Model Management?

Model Management is a universal model loader system that:

- **Loads Models**: Automatically loads any HuggingFace model
- **Inspects Capabilities**: Detects model type, labels, and taxonomy
- **Manages Registry**: Stores model metadata persistently
- **Tests Models**: Provides interface for testing with custom text
- **Adapts Outputs**: Converts different model formats to unified structure

## Key Features

### 📋 Model List

View and manage all registered models in your system.

**Features**:
- **Filter by Task**: text-classification, token-classification
- **Filter by Language**: English, Dutch, Persian, or all
- **Filter by Taxonomy**: GoEmotions, sentiment, custom
- **View Details**: Model ID, description, labels, usage statistics
- **Edit Models**: Modify names, descriptions, labels, taxonomy
- **Remove Models**: Delete models from registry

**Model Information Displayed**:
- Model ID (HuggingFace identifier)
- Display Name (human-readable)
- Description (what the model does)
- Task Type (classification type)
- Taxonomy (emotion/sentiment system)
- Number of Labels (output categories)
- Languages (supported languages)
- Tags (categorization)
- Use Count (how many times used)
- Last Used (timestamp)

### ➕ Add Model

Add new models from HuggingFace Hub with automatic inspection.

**Process**:
1. Enter HuggingFace Model ID
2. Provide display name and description
3. Add tags for categorization
4. System automatically:
   - Validates model exists
   - Loads model and tokenizer
   - Detects capabilities
   - Extracts labels
   - Identifies taxonomy
   - Detects supported languages
   - Creates metadata entry

**Supported Model Types**:
- **Emotion Classification**: GoEmotions, EmoRoBERTa, etc.
- **Sentiment Analysis**: Binary, ternary, 5-star ratings
- **Custom Classification**: Any text classification model

### 🧪 Test Model

Test registered models with custom text.

**Features**:
- Select any registered model
- View model information
- Enter test text
- Adjust top-k predictions
- View results with scores
- Export predictions as JSON
- Usage tracking (updates use count)

## How to Use

### Adding a New Model

#### Step 1: Find a Model

Browse HuggingFace Hub for models:
- [Emotion Models](https://huggingface.co/models?pipeline_tag=text-classification&search=emotion)
- [Sentiment Models](https://huggingface.co/models?pipeline_tag=text-classification&search=sentiment)

**Popular Models**:
- `j-hartmann/emotion-english-distilroberta-base` - English emotions (7 categories)
- `cardiffnlp/twitter-roberta-base-emotion` - Twitter emotions (4 categories)
- `nlptown/bert-base-multilingual-uncased-sentiment` - Multilingual sentiment (5 stars)
- `distilbert-base-uncased-finetuned-sst-2-english` - Binary sentiment

#### Step 2: Enter Model Information

Navigate to **"➕ Add Model"** tab and fill in:

**Required**:
- **HuggingFace Model ID**: Full model identifier (e.g., `j-hartmann/emotion-english-distilroberta-base`)

**Optional but Recommended**:
- **Display Name**: Human-readable name (e.g., "Emotion English DistilRoBERTa")
- **Description**: What the model does (e.g., "7-class emotion classification for English text")
- **Tags**: Categorization tags (e.g., "emotion, english, roberta")

#### Step 3: Load and Inspect

Click **"🔍 Load and Add Model"** button.

The system will:
1. Validate model exists on HuggingFace
2. Download and load model + tokenizer
3. Inspect model configuration
4. Detect capabilities automatically
5. Extract label mappings
6. Identify taxonomy type
7. Detect supported languages
8. Display detected information

**Detected Information**:
- Task Type (e.g., text-classification)
- Number of Labels (e.g., 7)
- Taxonomy (e.g., emotion, sentiment)
- Architecture (e.g., distilroberta)
- Max Length (e.g., 512 tokens)
- Languages (e.g., English)
- Label Mappings (index → label name)

#### Step 4: Review and Confirm

Review the detected information:
- Check labels are correct
- Verify taxonomy classification
- Confirm language support

The model is automatically added to the registry if validation succeeds.

### Editing a Model

#### Step 1: Find the Model

Navigate to **"📋 Model List"** tab and locate your model.

#### Step 2: Click Edit

Click the **"✏️ Edit"** button in the model's expander.

#### Step 3: Modify Details

Edit any of the following:
- **Display Name**: Change the human-readable name
- **Description**: Update the description
- **Taxonomy**: Change classification (goemotions, sentiment, custom, star_rating, binary)
- **Languages**: Add or remove supported languages (comma-separated)
- **Tags**: Update categorization tags (comma-separated)
- **Labels**: Edit label mappings (format: `index: label_name`)

**Label Editing**:
```
0: anger
1: joy
2: sadness
3: fear
4: surprise
5: love
6: neutral
```

#### Step 4: Save Changes

Click **"💾 Save Changes"** to persist modifications.

**Note**: Changes are saved to `config/models_registry.json` and persist across sessions.

### Testing a Model

#### Step 1: Select Model

Navigate to **"🧪 Test Model"** tab and select a model from the dropdown.

#### Step 2: View Model Info

Expand **"ℹ️ Model Information"** to see:
- Model ID
- Task type
- Taxonomy
- Number of labels
- Usage count

#### Step 3: Enter Test Text

Type or paste text in the input area.

**Tips**:
- Use text in the model's supported language
- Keep text concise (1-3 sentences)
- Ensure proper grammar and spelling

#### Step 4: Configure Analysis

Adjust **Top K** slider (1-10) to control how many predictions to show.

#### Step 5: Analyze

Click **"🔍 Analyze"** button.

Results display:
- **Top Predictions**: Ranked by confidence with progress bars
- **All Predictions**: Complete list with scores
- **Export**: Download as JSON

### Removing a Model

#### Step 1: Find the Model

Navigate to **"📋 Model List"** tab and locate the model.

#### Step 2: Click Remove

Click the **"🗑️ Remove"** button in the model's expander.

#### Step 3: Confirm

The model is immediately removed from the registry.

**Note**: This only removes the registry entry, not the cached model files.

## Model Taxonomy Types

### GoEmotions
28 fine-grained emotions organized into 4 sentiment groups:
- Positive (12 emotions)
- Negative (11 emotions)
- Ambiguous (4 emotions)
- Neutral (1 emotion)

### Sentiment
General sentiment classification:
- **Binary**: positive, negative
- **Ternary**: positive, neutral, negative
- **5-Star**: 1 star, 2 stars, 3 stars, 4 stars, 5 stars

### Custom
Any other classification scheme not matching standard taxonomies.

## Universal Adapter System

The system automatically adapts different model output formats:

### Star Rating Models
**Input**: 5 classes (1-5 stars)
**Output**: Converted to sentiment + emotion scores

Example:
```
5 stars → joy (high), optimism (medium)
1 star → anger (medium), disappointment (high)
```

### Binary Sentiment Models
**Input**: 2 classes (positive, negative)
**Output**: Expanded to emotion categories

Example:
```
positive → joy (high), love (medium), optimism (medium)
negative → anger (medium), sadness (medium), disappointment (high)
```

### Ternary Sentiment Models
**Input**: 3 classes (positive, neutral, negative)
**Output**: Mapped to emotion categories

Example:
```
positive → joy, love, optimism
neutral → neutral, realization
negative → anger, sadness, disappointment
```

## Registry Storage

Models are stored in `config/models_registry.json`:

```json
{
  "models": [
    {
      "model_id": "j-hartmann/emotion-english-distilroberta-base",
      "name": "Emotion English DistilRoBERTa",
      "description": "7-class emotion classification",
      "task_type": "text-classification",
      "language": ["english"],
      "num_labels": 7,
      "label_map": {
        "0": "anger",
        "1": "joy",
        "2": "sadness",
        ...
      },
      "taxonomy": "emotion",
      "tags": ["emotion", "english", "roberta"],
      "use_count": 5,
      "last_used": "2025-01-XX 10:30:00"
    }
  ]
}
```

## Integration with Analysis

Models added via Model Management automatically appear in:

1. **Analysis Tab**: Model dropdown for language-specific models
2. **Samples Tab**: Model selection for sample text analysis
3. **Model Registry**: Available for programmatic use

**Access Pattern**:
- Models with `registry:` prefix in model key
- Automatically loaded from registry
- Cached for performance

## Use Cases

### Research Projects

**Comparing Models**:
- Add multiple models for same task
- Test with same text across models
- Compare prediction differences
- Export results for analysis

**Custom Taxonomies**:
- Add domain-specific models
- Edit labels to match research needs
- Track usage for reproducibility

### Production Systems

**Model Evaluation**:
- Test candidate models before deployment
- Compare performance on real data
- Track usage statistics

**Multi-Model Systems**:
- Use different models for different languages
- Ensemble predictions from multiple models
- A/B testing with model variants

### Educational Use

**Learning About Models**:
- Explore different model architectures
- Compare classification schemes
- Understand label mappings

**Experimentation**:
- Try various HuggingFace models
- Test on different text types
- Learn about model capabilities

## Tips & Best Practices

### Adding Models

1. **Verify Model Type**: Ensure it's a text-classification model
2. **Check Language**: Confirm language matches your needs
3. **Review Labels**: Inspect extracted labels for correctness
4. **Add Description**: Write clear descriptions for future reference
5. **Use Tags**: Tag models for easy filtering

### Testing Models

1. **Use Appropriate Text**: Match text to model's training domain
2. **Test Multiple Examples**: Don't rely on single test
3. **Compare Top-K**: Look at multiple predictions, not just top-1
4. **Export Results**: Save predictions for comparison

### Managing Registry

1. **Remove Unused Models**: Keep registry clean
2. **Update Descriptions**: Keep information current
3. **Backup Registry**: Save `models_registry.json` periodically
4. **Monitor Usage**: Track which models are most used

## Limitations

### Current Limitations

- **Model Types**: Only text-classification models supported
- **Languages**: Auto-detection may miss some languages
- **Taxonomy**: Detection is heuristic-based, may need manual correction
- **Storage**: Models cached locally (can use significant disk space)

### Known Issues

- Some models may have tokenizer compatibility issues
- Label extraction may fail for non-standard configurations
- Very large models (>1GB) may be slow to load
- Persian/Arabic models may need special handling

## Technical Details

### Components

**ModelRegistry**:
- Stores model metadata
- Provides CRUD operations
- Persists to JSON file
- Thread-safe operations

**UniversalModelLoader**:
- Loads HuggingFace models
- Handles tokenizer variants
- Validates model compatibility
- Caches loaded models

**ModelInspector**:
- Detects model capabilities
- Extracts label mappings
- Identifies taxonomy
- Determines languages

**UniversalAdapter**:
- Provides unified prediction interface
- Converts model outputs
- Handles different formats
- Returns standardized results

### Performance

**Model Loading**:
- First load: 5-15 seconds (depends on model size)
- Cached load: <1 second
- Memory: 200MB-2GB per model

**Prediction**:
- Single text: 0.1-0.5 seconds
- Batch (10 texts): 0.5-2 seconds

## Troubleshooting

### Model Won't Load

**Issue**: "Model validation failed"
- **Solution**: Check model ID is correct, model exists on HuggingFace

**Issue**: "Tokenizer not found"
- **Solution**: Some models need special tokenizer handling, try different model

**Issue**: "Out of memory"
- **Solution**: Close other applications, try smaller model

### Labels Not Detected

**Issue**: "No labels found"
- **Solution**: Manually edit model after adding, specify labels

**Issue**: "Wrong labels detected"
- **Solution**: Use Edit feature to correct label mappings

### Predictions Incorrect

**Issue**: "Unexpected results"
- **Solution**: Check text language matches model, verify labels are correct

**Issue**: "Low confidence scores"
- **Solution**: Model may not be suitable for text type, try different model

## Support

For issues or questions:
- Check the [Universal Model Loader Guide](../../guides/universal-model-loader.md)
- Review the [Model Management Implementation](../../guides/phase2-streamlit-integration-complete.md)
- Consult HuggingFace model documentation

---

**Last Updated**: 2025-01-XX
**Version**: 0.2.0
**Feature**: Model Management

