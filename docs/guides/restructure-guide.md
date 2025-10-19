# Code Restructure Guide

## Overview

The Bahar codebase has been reorganized into a structured package to support multiple datasets and analysis methods.

## New Directory Structure

```
bahar/
├── bahar/                          # Main package
│   ├── __init__.py                 # Package exports
│   ├── datasets/                   # Dataset modules
│   │   ├── __init__.py
│   │   └── goemotions/             # GoEmotions dataset
│   │       ├── __init__.py
│   │       ├── taxonomy.py         # Emotion taxonomy
│   │       ├── classifier.py       # GoEmotions classifier
│   │       ├── result.py           # Result classes
│   │       └── samples.py          # Sample texts
│   ├── analyzers/                  # Analysis modules
│   │   ├── __init__.py
│   │   ├── emotion_analyzer.py     # Unified emotion analyzer
│   │   ├── linguistic_analyzer.py  # Linguistic analysis
│   │   ├── linguistic_samples.py   # Linguistic samples
│   │   └── enhanced_analyzer.py    # Combined analyzer
│   ├── cli/                        # Command-line tools
│   │   ├── __init__.py
│   │   ├── classify_basic.py       # Basic CLI
│   │   ├── classify_enhanced.py    # Enhanced CLI
│   │   └── test_categories.py      # Category testing
│   ├── demos/                      # Demo scripts
│   │   ├── __init__.py
│   │   ├── demo_basic.py           # Basic demo
│   │   └── demo_enhanced.py        # Enhanced demo
│   └── utils/                      # Utility functions
│       └── __init__.py
├── main.py                         # Backward compatible wrapper
├── classify_text.py                # Backward compatible wrapper
├── classify_enhanced.py            # Backward compatible wrapper
├── emotion_classification_demo.ipynb  # Jupyter notebook
├── pyproject.toml                  # Project configuration
└── README.md                       # Documentation
```

## Migration Guide

### Old vs. New Imports

#### Emotion Classification

**Old:**
```python
from emotion_classifier import MultilingualEmotionClassifier, EmotionResult
```

**New:**
```python
from bahar.analyzers import EmotionAnalyzer
from bahar.datasets.goemotions import GoEmotionsClassifier
from bahar.datasets.goemotions.result import EmotionResult
```

#### Linguistic Analysis

**Old:**
```python
from linguistic_analyzer import LinguisticAnalyzer, LinguisticFeatures
```

**New:**
```python
from bahar.analyzers import LinguisticAnalyzer
from bahar.analyzers.linguistic_analyzer import LinguisticFeatures
```

#### Enhanced Analysis

**Old:**
```python
from enhanced_classifier import EnhancedEmotionClassifier
```

**New:**
```python
from bahar.analyzers import EnhancedAnalyzer
```

#### GoEmotions Taxonomy

**Old:**
```python
from emotion_classifier import GOEMOTIONS_EMOTIONS, EMOTION_GROUPS
```

**New:**
```python
from bahar.datasets.goemotions import GOEMOTIONS_EMOTIONS, EMOTION_GROUPS
```

#### Sample Texts

**Old:**
```python
from sample_texts import SAMPLE_TEXTS
```

**New:**
```python
from bahar.datasets.goemotions import SAMPLE_TEXTS
```

### Usage Examples

#### Basic Emotion Analysis

```python
from bahar.analyzers import EmotionAnalyzer

# Initialize analyzer
analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

# Analyze text
result = analyzer.analyze("I'm so happy!", top_k=3)
print(result.get_top_emotions())
```

#### Enhanced Analysis (Emotion + Linguistics)

```python
from bahar.analyzers import EnhancedAnalyzer
from bahar.analyzers.enhanced_analyzer import format_enhanced_output

# Initialize analyzer
analyzer = EnhancedAnalyzer(emotion_dataset="goemotions")
analyzer.load_model()

# Analyze text
result = analyzer.analyze("I hereby formally request your assistance.", top_k=3)
print(format_enhanced_output(result))

# Access specific features
print(f"Formality: {result.linguistic_features.formality}")
print(f"Tone: {result.linguistic_features.tone}")
```

#### Direct GoEmotions Classifier

```python
from bahar.datasets.goemotions import GoEmotionsClassifier
from bahar.datasets.goemotions.result import format_emotion_output

# Initialize classifier
classifier = GoEmotionsClassifier()
classifier.load_model()

# Classify
result = classifier.predict("Your text here", top_k=3)
print(format_emotion_output(result))
```

## Backward Compatibility

The root-level scripts (`main.py`, `classify_text.py`, `classify_enhanced.py`) have been updated to use the new package structure but maintain the same CLI interface.

**All existing commands still work:**

```bash
# These still work exactly as before
python main.py
python classify_text.py "Your text"
python classify_enhanced.py "Your text" --export-json
```

## Benefits of New Structure

### 1. **Modularity**
- Clear separation of concerns
- Easy to add new datasets
- Independent module testing

### 2. **Extensibility**
- Add new emotion datasets alongside GoEmotions
- Add new analysis methods easily
- Support multiple models per dataset

### 3. **Maintainability**
- Organized code structure
- Clear dependencies
- Better code navigation

### 4. **Academic Research**
- Easy to compare different datasets
- Structured for experimental workflows
- Clear separation of data and methods

## Adding New Datasets

To add a new emotion dataset (e.g., EmoBank, ISEAR):

1. Create directory: `bahar/datasets/your_dataset/`
2. Add files:
   - `__init__.py` - Package exports
   - `taxonomy.py` - Emotion categories
   - `classifier.py` - Classifier implementation
   - `result.py` - Result classes
   - `samples.py` - Sample texts

3. Update `bahar/analyzers/emotion_analyzer.py`:
```python
if dataset == "your_dataset":
    from bahar.datasets.your_dataset import YourClassifier
    self.classifier = YourClassifier(model_name)
```

4. Use it:
```python
analyzer = EmotionAnalyzer(dataset="your_dataset")
```

## Testing the New Structure

```bash
# Test basic emotion analysis
python -c "from bahar.analyzers import EmotionAnalyzer; print('✓ EmotionAnalyzer')"

# Test GoEmotions
python -c "from bahar.datasets.goemotions import GoEmotionsClassifier; print('✓ GoEmotions')"

# Test linguistic analysis
python -c "from bahar.analyzers import LinguisticAnalyzer; print('✓ LinguisticAnalyzer')"

# Test enhanced analysis
python -c "from bahar.analyzers import EnhancedAnalyzer; print('✓ EnhancedAnalyzer')"

# Run backward compatible scripts
python main.py
python classify_text.py "I'm happy!"
```

## File Mapping

| Old Location | New Location |
|--------------|--------------|
| `emotion_classifier.py` | `bahar/datasets/goemotions/classifier.py` + `result.py` + `taxonomy.py` |
| `linguistic_analyzer.py` | `bahar/analyzers/linguistic_analyzer.py` |
| `enhanced_classifier.py` | `bahar/analyzers/enhanced_analyzer.py` |
| `sample_texts.py` | `bahar/datasets/goemotions/samples.py` |
| `linguistic_samples.py` | `bahar/analyzers/linguistic_samples.py` |
| `main.py` | `bahar/demos/demo_basic.py` (+ wrapper) |
| `demo_enhanced.py` | `bahar/demos/demo_enhanced.py` (+ wrapper) |
| `classify_text.py` | `bahar/cli/classify_basic.py` (+ wrapper) |
| `classify_enhanced.py` | `bahar/cli/classify_enhanced.py` (+ wrapper) |
| `test_linguistic_categories.py` | `bahar/cli/test_categories.py` |

## Next Steps

1. **Test all functionality** with new imports
2. **Update Jupyter notebook** to use new imports
3. **Add new datasets** as needed
4. **Extend analyzers** with new methods
5. **Create tests** for each module

## Questions?

See the main README.md for usage examples and documentation.

