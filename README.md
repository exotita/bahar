# Bahar - Multilingual Emotion Classification

A simple emotion classification system using the GoEmotions taxonomy for Dutch, Persian, and English text.

## About GoEmotions

This project is based on Google Research's [GoEmotions dataset](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/), which provides fine-grained emotion classification with 27 emotion categories plus neutral:

### Emotion Categories

**Positive emotions (12):**
- admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief

**Negative emotions (11):**
- anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness

**Ambiguous emotions (4):**
- confusion, curiosity, realization, surprise

**Neutral (1):**
- neutral

## Features

### Emotion Analysis (GoEmotions)
- Multilingual emotion classification for Dutch, Persian, and English
- 28 fine-grained emotion categories
- Based on pre-trained GoEmotions model
- Returns top-k emotions with confidence scores
- Automatic sentiment grouping (positive/negative/ambiguous/neutral)

### Linguistic Analysis (Academic Dimensions)
- **Formality Detection**: formal, colloquial, neutral
- **Tone Analysis**: friendly, rough, serious, kind, neutral
- **Intensity Measurement**: high, medium, low emotional intensity
- **Communication Style**: direct, indirect, assertive, passive
- **48 multilingual samples** across 16 categories (English, Dutch, Persian)
- Suitable for academic linguistic research
- Export-ready format for research data collection

## Installation

### Option 1: Docker (Recommended for Production)

**Quick Start:**
```bash
# Development mode
./docker-start.sh

# Production mode with Nginx
./docker-start.sh --prod

# Access the app
open http://localhost:8501  # Development
open http://localhost        # Production
```

**Features:**
- ✅ All dependencies pre-installed
- ✅ Models cached in `volumes/` directory
- ✅ Easy backup and migration
- ✅ Production-ready with Nginx
- ✅ Auto-restart on failure

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for complete documentation.

### Option 2: Local Development

1. Ensure Python 3.12 is installed (check `.python-version`)

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install transformers torch rich
```

Or sync from `pyproject.toml`:
```bash
uv sync
```

**Dependencies:**
- `transformers>=4.57.0` - HuggingFace transformers
- `torch>=2.9.0` - PyTorch for model inference
- `rich>=14.2.0` - Beautiful terminal output

## Usage

### Run the basic demo:

```bash
python main.py
```

The demo will:
1. Load the pre-trained GoEmotions model (downloads ~400MB on first run)
2. Process sample texts in English, Dutch, and Persian
3. Display top 3 emotions with confidence scores for each text
4. Show sentiment grouping and expected emotions

### Run the enhanced demo (with linguistic analysis):

```bash
python demo_enhanced.py
```

The enhanced demo provides:
1. All emotion analysis features
2. Linguistic formality detection
3. Tone and intensity analysis
4. Communication style identification
5. Academic export format examples

### Classify custom text (basic):

```bash
python classify_text.py "Your text here"
```

With custom top-k:
```bash
python classify_text.py "I'm so excited!" --top-k 5
```

### Classify with enhanced analysis (emotion + linguistics):

```bash
python classify_enhanced.py "Your text here"
```

Export as JSON for research:
```bash
python classify_enhanced.py "Your text" --export-json
```

Examples:
```bash
# English - basic
python classify_text.py "Thank you so much for your help!"

# English - enhanced with linguistic analysis
python classify_enhanced.py "I hereby formally request your assistance."

# Dutch
python classify_enhanced.py "Ik ben zo blij met dit nieuws!"

# Persian
python classify_enhanced.py "من از این خبر خیلی خوشحالم!"

# Export for academic research
python classify_enhanced.py "Your research text" --export-json > results.json

# Test all linguistic categories
python test_linguistic_categories.py
```

### Use in your code (basic):

```python
from bahar import EmotionAnalyzer
from bahar.datasets.goemotions.result import format_emotion_output

# Initialize analyzer
analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

# Analyze a text
result = analyzer.analyze("I'm so excited about this!", top_k=3)

# Display results
print(format_emotion_output(result))

# Access raw data
top_emotions = result.get_top_emotions()
sentiment = result.get_sentiment_group()
```

### Use enhanced analyzer (emotion + linguistics):

```python
from bahar import EnhancedAnalyzer
from bahar.analyzers.enhanced_analyzer import (
    format_enhanced_output,
    export_to_academic_format,
)

# Initialize enhanced analyzer
analyzer = EnhancedAnalyzer(emotion_dataset="goemotions")
analyzer.load_model()

# Analyze text
result = analyzer.analyze("I hereby formally request your assistance.", top_k=3)

# Display comprehensive analysis
print(format_enhanced_output(result))

# Access specific dimensions
print(f"Formality: {result.linguistic_features.formality}")
print(f"Tone: {result.linguistic_features.tone}")
print(f"Intensity: {result.linguistic_features.intensity}")
print(f"Style: {result.linguistic_features.communication_style}")

# Export for academic research
academic_data = export_to_academic_format(result)
# academic_data is a dict ready for CSV/JSON export
```

## Project Structure

```
bahar/
├── main.py                              # Basic demo (emotions only)
├── demo_enhanced.py                     # Enhanced demo (emotions + linguistics)
├── classify_text.py                     # CLI for basic emotion classification
├── classify_enhanced.py                 # CLI for enhanced analysis
├── test_linguistic_categories.py        # Test all linguistic categories
├── emotion_classifier.py                # Core emotion classifier (GoEmotions)
├── linguistic_analyzer.py               # Linguistic dimension analyzer
├── enhanced_classifier.py               # Combined emotion + linguistic classifier
├── sample_texts.py                      # Basic sample texts in 3 languages
├── linguistic_samples.py                # 48 samples across 16 categories
├── emotion_classification_demo.ipynb    # Jupyter notebook demo
├── pyproject.toml                       # Project configuration
├── .python-version                      # Python version (3.12)
├── .gitignore                           # Git ignore rules
├── README.md                            # This file
├── QUICK_START.md                       # Quick reference guide
├── LINGUISTIC_CATEGORIES.md             # Linguistic categories documentation
└── IMPLEMENTATION_SUMMARY.md            # Technical details
```

## Model Information

By default, this project uses `monologg/bert-base-cased-goemotions-original`, which is fine-tuned on the GoEmotions dataset. This model works best for English text.

For better multilingual support (Dutch, Persian, and other languages), the classifier can be extended to use:
- `bert-base-multilingual-cased` (requires fine-tuning on GoEmotions)
- `xlm-roberta-base` (requires fine-tuning on GoEmotions)

## Limitations

- The pre-trained model is optimized for English text
- Dutch and Persian classification may be less accurate without multilingual fine-tuning
- Emotions are subtle and context-dependent; results are probabilistic
- Maximum input length is 512 tokens

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### User Guides
- **[Documentation Index](docs/README.md)** - Complete documentation overview
- **[Quick Start Guide](docs/guides/quick-start.md)** - Get started in 5 minutes
- **[Installation Guide](docs/guides/installation.md)** - Detailed setup instructions
- **[GoEmotions Documentation](docs/goemotions/README.md)** - Dataset details and taxonomy
- **[Linguistic Analysis Guide](docs/guides/linguistic-analysis.md)** - Academic dimensions
- **[Rich Output Guide](docs/guides/rich-output.md)** - Beautiful terminal output
- **[Adding Datasets Guide](docs/guides/adding-datasets.md)** - Extend with new datasets
- **[Migration Guide](docs/guides/migration.md)** - Upgrade from old structure

### Developer Resources
- **[.cursorrules](.cursorrules)** - Project structure, coding standards, and development guidelines
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Complete development workflow and best practices
- **[Cursor Setup Guide](docs/guides/cursor-setup.md)** - Cursor AI configuration verification
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

### Project History & Migration
- **[Restructure Guide](docs/guides/restructure-guide.md)** - Code reorganization details
- **[Migration Complete](docs/guides/migration-complete.md)** - Migration summary

## References

- [GoEmotions: A Dataset of Fine-Grained Emotions](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)
- [GoEmotions Dataset on GitHub](https://github.com/google-research/google-research/tree/master/goemotions)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Rich Library](https://rich.readthedocs.io/)

## License

This project uses the GoEmotions dataset and pre-trained models. Please refer to the original dataset license for usage terms.

