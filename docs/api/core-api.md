# Core API Reference

## EmotionAnalyzer

The main class for emotion classification using various models and datasets.

### Constructor

```python
EmotionAnalyzer(
    language: str | None = None,
    model_key: str | None = None,
    model_name: str | None = None,
    auto_detect_language: bool = True
)
```

**Parameters:**

- `language` (str | None): Language code (`"english"`, `"dutch"`, `"persian"`)
- `model_key` (str | None): Model identifier (e.g., `"goemotions"`, `"sentiment"`)
- `model_name` (str | None): Explicit HuggingFace model name
- `auto_detect_language` (bool): Enable automatic language detection

**Example:**

```python
# Using language and model key
analyzer = EmotionAnalyzer(language="english", model_key="goemotions")

# Using explicit model name
analyzer = EmotionAnalyzer(model_name="monologg/bert-base-cased-goemotions-original")

# Auto-detect language
analyzer = EmotionAnalyzer(auto_detect_language=True)
```

### Methods

#### `load_model()`

Load the emotion classification model into memory.

```python
analyzer.load_model()
```

**Raises:**
- `RuntimeError`: If model loading fails

#### `analyze(text: str, top_k: int = 3) -> EmotionResult`

Analyze emotions in a single text.

**Parameters:**
- `text` (str): Input text to analyze
- `top_k` (int): Number of top emotions to return (default: 3)

**Returns:**
- `EmotionResult`: Analysis results with emotion scores

**Example:**

```python
result = analyzer.analyze("I'm so happy!", top_k=5)

# Get top emotions
for emotion, score in result.get_top_emotions():
    print(f"{emotion}: {score:.3f}")

# Get all emotions
all_emotions = result.emotions  # dict[str, float]

# Get sentiment group
sentiment = result.get_sentiment_group()  # "positive", "negative", "ambiguous", "neutral"
```

#### `analyze_batch(texts: list[str], top_k: int = 3) -> list[EmotionResult]`

Analyze emotions in multiple texts.

**Parameters:**
- `texts` (list[str]): List of texts to analyze
- `top_k` (int): Number of top emotions per text

**Returns:**
- `list[EmotionResult]`: List of analysis results

**Example:**

```python
texts = ["I'm happy!", "This is sad.", "I'm confused."]
results = analyzer.analyze_batch(texts, top_k=3)

for text, result in zip(texts, results):
    print(f"{text}: {result.get_top_emotions()[0]}")
```

#### `get_model_info() -> dict[str, str]`

Get information about the loaded model.

**Returns:**
- `dict[str, str]`: Model metadata

**Example:**

```python
info = analyzer.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Dataset: {info['dataset']}")
```

---

## EnhancedAnalyzer

Combines emotion analysis with linguistic dimension analysis.

### Constructor

```python
EnhancedAnalyzer(
    language: str | None = None,
    model_key: str | None = None,
    emotion_dataset: str = "goemotions"
)
```

**Parameters:**

- `language` (str | None): Language code
- `model_key` (str | None): Model identifier
- `emotion_dataset` (str): Dataset name (default: `"goemotions"`)

**Example:**

```python
analyzer = EnhancedAnalyzer(
    language="english",
    model_key="goemotions"
)
```

### Methods

#### `load_model()`

Load both emotion and linguistic analysis models.

```python
analyzer.load_model()
```

#### `analyze(text: str, top_k: int = 3) -> EnhancedAnalysisResult`

Perform comprehensive analysis including emotions and linguistics.

**Parameters:**
- `text` (str): Input text
- `top_k` (int): Number of top emotions

**Returns:**
- `EnhancedAnalysisResult`: Combined results

**Example:**

```python
result = analyzer.analyze("I'm extremely disappointed!", top_k=3)

# Emotion results
emotion_result = result.emotion_result
top_emotions = emotion_result.get_top_emotions()

# Linguistic features
features = result.linguistic_features
print(f"Formality: {features.formality}")
print(f"Tone: {features.tone}")
print(f"Intensity: {features.intensity}")
print(f"Style: {features.communication_style}")
```

#### `analyze_batch(texts: list[str], top_k: int = 3) -> list[EnhancedAnalysisResult]`

Analyze multiple texts with full analysis.

**Parameters:**
- `texts` (list[str]): List of texts
- `top_k` (int): Number of top emotions per text

**Returns:**
- `list[EnhancedAnalysisResult]`: List of results

---

## LinguisticAnalyzer

Analyzes linguistic dimensions without emotion classification.

### Constructor

```python
LinguisticAnalyzer()
```

### Methods

#### `analyze(text: str) -> LinguisticFeatures`

Analyze linguistic dimensions of text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `LinguisticFeatures`: Linguistic analysis results

**Example:**

```python
from bahar.analyzers.linguistic_analyzer import LinguisticAnalyzer

analyzer = LinguisticAnalyzer()
features = analyzer.analyze("I am extremely disappointed with this service.")

print(f"Formality: {features.formality} ({features.formality_score:.2%})")
print(f"Tone: {features.tone} ({features.tone_score:.2%})")
print(f"Intensity: {features.intensity} ({features.intensity_score:.2%})")
print(f"Style: {features.communication_style} ({features.style_score:.2%})")
```

---

## Utility Functions

### Language Detection

```python
from bahar.utils.language_models import detect_language

language = detect_language("This is English text")
# Returns: "english"

language = detect_language("Dit is Nederlandse tekst")
# Returns: "dutch"

language = detect_language("این متن فارسی است")
# Returns: "persian"
```

### Available Models

```python
from bahar.utils.language_models import (
    get_available_models,
    get_supported_languages
)

# Get all supported languages
languages = get_supported_languages()
# Returns: ["english", "dutch", "persian"]

# Get models for a language
models = get_available_models("english")
# Returns: dict of model_key -> model_name

# Include registry models
models = get_available_models("english", include_registry=True)
```

### Model Names

```python
from bahar.utils.language_models import get_model_name

# Get model name by language and key
model_name = get_model_name("english", "goemotions")
# Returns: "monologg/bert-base-cased-goemotions-original"

model_name = get_model_name("dutch", "sentiment")
# Returns: "nlptown/bert-base-multilingual-uncased-sentiment"
```

---

## Export Functions

### Academic Format Export

```python
from bahar.analyzers.enhanced_analyzer import export_to_academic_format

# Analyze text
result = analyzer.analyze("Your text here")

# Export to structured format
academic_data = export_to_academic_format(result)

# Structure:
{
    "text": "...",
    "timestamp": "...",
    "language": "...",
    "model": "...",
    "top_emotion_1": "...",
    "top_emotion_1_score": 0.xxx,
    "top_emotion_2": "...",
    "top_emotion_2_score": 0.xxx,
    "top_emotion_3": "...",
    "top_emotion_3_score": 0.xxx,
    "sentiment": "...",
    "formality": "...",
    "formality_score": 0.xxx,
    "tone": "...",
    "tone_score": 0.xxx,
    "intensity": "...",
    "intensity_score": 0.xxx,
    "communication_style": "...",
    "style_score": 0.xxx
}
```

---

## Constants

### GoEmotions Emotions

```python
from bahar.datasets.goemotions import GOEMOTIONS_EMOTIONS

# List of all 28 emotions
emotions = GOEMOTIONS_EMOTIONS
# ['admiration', 'amusement', 'anger', ...]
```

### Emotion Groups

```python
from bahar.datasets.goemotions import EMOTION_GROUPS

# Grouped by sentiment
groups = EMOTION_GROUPS
# {
#     "positive": [...],
#     "negative": [...],
#     "ambiguous": [...],
#     "neutral": [...]
# }
```

---

## Type Hints

All API functions include complete type hints for better IDE support:

```python
from bahar import EmotionAnalyzer
from bahar.datasets.goemotions.result import EmotionResult

analyzer: EmotionAnalyzer = EmotionAnalyzer(language="english")
result: EmotionResult = analyzer.analyze("text")
top_emotions: list[tuple[str, float]] = result.get_top_emotions()
```

---

## Thread Safety

**Note:** Model loading is not thread-safe. Load models in the main thread before using them in worker threads. Once loaded, analyzers can be used safely across threads for read-only operations.

```python
# Good: Load in main thread
analyzer = EmotionAnalyzer(language="english")
analyzer.load_model()

# Then use in threads
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(analyzer.analyze, texts)
```

---

## See Also

- [Model Management API](./model-management.md)
- [Data Structures](./data-structures.md)
- [Examples](./examples.md)

