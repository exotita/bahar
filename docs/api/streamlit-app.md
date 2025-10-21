# Streamlit Web Application Documentation

## Overview

The Bahar Streamlit app provides a user-friendly web interface for emotion and linguistic analysis. It includes interactive analysis, sample testing, model management, and configuration options.

## Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the app
streamlit run app.py

# The app will open in your browser at http://localhost:8501
```

## Application Structure

### Main Tabs

1. **🎯 Analysis** - Interactive text analysis
2. **📚 Samples** - Pre-loaded sample texts
3. **🤖 Model Management** - Dynamic model loading
4. **⚙️ Configuration** - Customize settings
5. **📖 Documentation** - Help and API reference

---

## Tab 1: Analysis

### Features

- **Analysis Type Selection**
  - Basic Emotion: Emotion classification only
  - Enhanced: Emotion + linguistic analysis

- **Language & Model Selection**
  - Choose from English, Dutch, Persian
  - Select specific models for each language
  - Includes both built-in and registry models

- **Top-K Configuration**
  - Adjust number of top emotions to display (1-10)

- **Real-time Analysis**
  - Enter text and get instant results
  - Visual emotion scores with progress bars
  - Sentiment classification
  - Linguistic dimensions (Enhanced mode)

### Usage

1. Select analysis type (Basic or Enhanced)
2. Choose language and model
3. Set top-K value
4. Enter or paste text
5. Click "🔍 Analyze"

### Results Display

**Basic Emotion Analysis:**
- Top emotions with scores and percentages
- Visual confidence bars
- Overall sentiment classification
- Expandable view of all emotion scores

**Enhanced Analysis:**
- All basic emotion features
- Linguistic dimensions:
  - Formality (formal, colloquial, neutral)
  - Tone (friendly, rough, serious, kind, neutral)
  - Intensity (high, medium, low)
  - Communication style (direct, indirect, assertive, passive)
- Export options (JSON/CSV)

---

## Tab 2: Samples

### Features

- Pre-loaded sample texts in multiple languages
- Organized by category (positive, negative, ambiguous)
- One-click analysis
- Language-specific samples

### Usage

1. Select language from dropdown
2. Browse available samples
3. Click "🔍 Analyze" on any sample
4. View results inline

### Sample Categories

- **Positive**: Happy, excited, grateful expressions
- **Negative**: Disappointed, angry, sad expressions
- **Ambiguous**: Confused, uncertain expressions

---

## Tab 3: Model Management

### Overview

Dynamic model management system for loading and testing HuggingFace models.

### Sub-tabs

#### 📋 Model List

**Features:**
- View all registered models
- Filter by task type, language, taxonomy
- Edit model details and labels
- Remove models
- View usage statistics

**Filters:**
- Task Type: text-classification, token-classification
- Language: english, dutch, persian, all
- Taxonomy: goemotions, sentiment, custom

**Model Information:**
- Model ID (HuggingFace identifier)
- Display name
- Description
- Task type and taxonomy
- Number of labels
- Supported languages
- Tags
- Use count and last used timestamp

**Edit Mode:**
- Update display name and description
- Modify taxonomy classification
- Change supported languages
- Edit tags
- **Edit label mappings** (index: label_name)
- Save changes persistently

#### ➕ Add Model

**Features:**
- Load any HuggingFace text-classification model
- Automatic capability detection
- Auto-extract labels and taxonomy
- Language detection
- Tag management

**Process:**
1. Enter HuggingFace model ID
2. Provide display name (optional)
3. Add description (optional)
4. Add tags (comma-separated)
5. Click "🔍 Load and Add Model"
6. System automatically:
   - Validates model
   - Loads model and tokenizer
   - Detects task type and architecture
   - Extracts label mappings
   - Identifies taxonomy
   - Detects supported languages

**Detected Information:**
- Task type (e.g., text-classification)
- Number of labels
- Taxonomy (goemotions, sentiment, custom, star_rating, binary)
- Architecture (BERT, RoBERTa, ALBERT, etc.)
- Max sequence length
- Supported languages

#### 🧪 Test Model

**Features:**
- Test registered models with custom text
- View top-K predictions
- See all prediction scores
- Export results (JSON)
- Track model usage

**Usage:**
1. Select model from dropdown
2. Enter test text
3. Set top-K value
4. Click "🔍 Analyze"
5. View results with confidence scores
6. Download results as JSON

---

## Tab 4: Configuration

### Overview

Customize taxonomy, emotion groups, and sample texts.

### Sub-tabs

#### Taxonomy

**Features:**
- View all emotions in the system
- Edit emotion list (one per line)
- Save custom taxonomy
- Reset to default (GoEmotions 28 emotions)

**Usage:**
1. Edit emotions in text area
2. Click "💾 Save Taxonomy"
3. Or click "🔄 Reset to Default"

#### Emotion Groups

**Features:**
- Organize emotions by sentiment
- Edit each group separately
- Four default groups:
  - Positive (12 emotions)
  - Negative (11 emotions)
  - Ambiguous (4 emotions)
  - Neutral (1 emotion)

**Usage:**
1. Expand emotion group
2. Edit emotions (one per line)
3. Click "Save {group}" or "💾 Save All Groups"
4. Or click "🔄 Reset to Default"

#### Samples

**Features:**
- Add new sample texts
- Edit existing samples (JSON format)
- Delete samples by language
- Organize by language and category

**Add New Sample:**
1. Enter language code
2. Enter category (e.g., positive, negative)
3. Enter sample text
4. Click "➕ Add Sample"

**Edit Samples:**
1. Select language
2. Edit JSON in text area
3. Click "💾 Save Changes"
4. Or click "🗑️ Delete All"

**Sample Format (JSON):**
```json
[
  {
    "text": "Sample text here",
    "category": "positive"
  }
]
```

---

## Tab 5: Documentation

### Sub-tabs

#### Overview

- About Bahar
- How to use the application
- Feature list
- Version information

#### Emotions

- Complete list of emotion categories
- Grouped by sentiment
- Color-coded display

#### Linguistic Dimensions

- Formality levels and definitions
- Tone types and characteristics
- Intensity levels
- Communication styles

#### API

- Links to detailed API documentation
- Code examples
- Quick reference

---

## Configuration Files

The application stores configuration in `config/` directory:

### Files

- `config/taxonomy.json` - Emotion taxonomy
- `config/emotion_groups.json` - Emotion sentiment groups
- `config/samples.json` - Sample texts by language
- `config/models_registry.json` - Registered models
- `config/logo.png` - Application logo

### Format

**taxonomy.json:**
```json
[
  "admiration",
  "amusement",
  "anger",
  ...
]
```

**emotion_groups.json:**
```json
{
  "positive": ["admiration", "amusement", ...],
  "negative": ["anger", "annoyance", ...],
  "ambiguous": ["confusion", "curiosity", ...],
  "neutral": ["neutral"]
}
```

**samples.json:**
```json
{
  "english": [
    {
      "text": "I'm so happy!",
      "category": "positive"
    }
  ],
  "dutch": [...],
  "persian": [...]
}
```

**models_registry.json:**
```json
{
  "model_id": {
    "model_id": "...",
    "name": "...",
    "description": "...",
    "task_type": "...",
    "language": [...],
    "num_labels": 0,
    "label_map": {...},
    "taxonomy": "...",
    "tags": [...],
    "use_count": 0,
    "last_used": null
  }
}
```

---

## Keyboard Shortcuts

- **Ctrl+Enter** / **Cmd+Enter**: Submit analysis (in text area)
- **Ctrl+R** / **Cmd+R**: Refresh page
- **Ctrl+S** / **Cmd+S**: Save (when editing configuration)

---

## Tips & Best Practices

### Performance

1. **Model Loading**: Models are cached after first load
2. **Batch Analysis**: Use samples tab for multiple texts
3. **Model Selection**: Choose appropriate models for your language

### Accuracy

1. **Language Selection**: Always select correct language
2. **Model Choice**: Different models have different strengths
3. **Text Length**: Optimal length is 50-200 words

### Configuration

1. **Backup**: Export configuration before making changes
2. **Testing**: Test changes with sample texts
3. **Reset**: Use reset buttons if something goes wrong

---

## Troubleshooting

### Common Issues

**Model Loading Fails:**
- Check internet connection (first-time download)
- Verify model ID is correct
- Check disk space for model cache

**Slow Performance:**
- Models are cached after first load
- Use appropriate model for your hardware
- Consider using smaller models for faster inference

**Incorrect Results:**
- Verify correct language is selected
- Try different models
- Check text formatting (remove special characters)

**Configuration Not Saving:**
- Check file permissions in `config/` directory
- Verify JSON format is valid
- Look for error messages in UI

---

## Advanced Features

### Custom Models

Add any HuggingFace text-classification model:

1. Go to Model Management → Add Model
2. Enter model ID (e.g., `cardiffnlp/twitter-roberta-base-emotion`)
3. System auto-detects capabilities
4. Model becomes available in Analysis tab

### Label Editing

Customize model labels:

1. Go to Model Management → Model List
2. Click "✏️ Edit" on any model
3. Edit labels in format: `index: label_name`
4. Save changes
5. Labels update across entire application

### Export & Integration

Export results for research:

1. Use Enhanced Analysis mode
2. Analyze text
3. Click "📥 Download JSON" or "📥 Download CSV"
4. Import into your analysis tools

---

## API Integration

The Streamlit app uses the same API as the Python package:

```python
# What the app does internally
from bahar import EmotionAnalyzer, EnhancedAnalyzer

# Load analyzer (cached)
@st.cache_resource
def load_analyzer(language, model_key):
    analyzer = EmotionAnalyzer(language=language, model_key=model_key)
    analyzer.load_model()
    return analyzer

# Analyze text
analyzer = load_analyzer("english", "goemotions")
result = analyzer.analyze(text, top_k=3)
```

You can use the same API in your own applications!

---

## See Also

- [Core API Reference](./core-api.md)
- [Model Management API](./model-management.md)
- [Quick Start Guide](../guides/quick-start.md)
- [Installation Guide](../guides/installation.md)

