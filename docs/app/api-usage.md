### 💻 API Usage

Learn how to use baarsh programmatically in your Python applications.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/baarsh/baarsh.git
cd baarsh

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Basic Emotion Analysis

```python
from bahar import EmotionAnalyzer

# Initialize analyzer
analyzer = EmotionAnalyzer(language="english", model_key="goemotions")

# Load model (required before analysis)
analyzer.load_model()

# Analyze text
result = analyzer.analyze("I'm so happy and excited!", top_k=3)

# Get top emotions
for emotion, score in result.get_top_emotions():
    print(f"{emotion}: {score:.3f}")

# Output:
# joy: 0.856
# excitement: 0.742
# optimism: 0.621

# Get sentiment
sentiment = result.get_sentiment_group()
print(f"Sentiment: {sentiment}")  # Output: positive
```

---

### Enhanced Analysis

Combine emotion detection with linguistic analysis:

```python
from bahar import EnhancedAnalyzer

# Initialize enhanced analyzer
analyzer = EnhancedAnalyzer(
    language="english",
    model_key="goemotions"
)

# Load model
analyzer.load_model()

# Analyze text
result = analyzer.analyze(
    "I'm extremely disappointed with this service!",
    top_k=3
)

# Access emotion results
print("=== Emotions ===")
for emotion, score in result.emotion_result.get_top_emotions():
    print(f"{emotion}: {score:.3f}")

# Access linguistic features
features = result.linguistic_features
print("\n=== Linguistic Dimensions ===")
print(f"Formality: {features.formality} ({features.formality_score:.1%})")
print(f"Tone: {features.tone} ({features.tone_score:.1%})")
print(f"Intensity: {features.intensity} ({features.intensity_score:.1%})")
print(f"Style: {features.communication_style} ({features.style_score:.1%})")
```

---

## Multilingual Support

### English

```python
from bahar import EmotionAnalyzer

analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
analyzer.load_model()

result = analyzer.analyze("This is wonderful!")
print(result.get_top_emotions()[0])  # ('joy', 0.xxx)
```

### Dutch

```python
analyzer = EmotionAnalyzer(language="dutch", model_key="sentiment")
analyzer.load_model()

result = analyzer.analyze("Dit is geweldig!")
print(result.get_sentiment_group())  # 'positive'
```

### Persian

```python
analyzer = EmotionAnalyzer(language="persian", model_key="sentiment")
analyzer.load_model()

result = analyzer.analyze("این عالی است!")
print(result.get_sentiment_group())  # 'positive'
```

---

## Batch Processing

Process multiple texts efficiently:

```python
from bahar import EmotionAnalyzer

analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
analyzer.load_model()

texts = [
    "I'm so happy!",
    "This is terrible.",
    "I'm not sure how I feel."
]

# Analyze all texts
results = analyzer.analyze_batch(texts, top_k=3)

# Process results
for text, result in zip(texts, results):
    top_emotion, score = result.get_top_emotions()[0]
    sentiment = result.get_sentiment_group()
    print(f"Text: {text}")
    print(f"Top Emotion: {top_emotion} ({score:.3f})")
    print(f"Sentiment: {sentiment}\n")
```

---

## Export Results

### Academic Format

Export structured data for research:

```python
from bahar import EnhancedAnalyzer
from bahar.analyzers.enhanced_analyzer import export_to_academic_format
import json

# Analyze text
analyzer = EnhancedAnalyzer(language="english", model_key="goemotions")
analyzer.load_model()
result = analyzer.analyze("Your text here", top_k=3)

# Export to academic format
academic_data = export_to_academic_format(result)

# Save as JSON
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(academic_data, f, indent=2, ensure_ascii=False)

# Save as CSV
import csv
with open("result.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=academic_data.keys())
    writer.writeheader()
    writer.writerow(academic_data)
```

### JSON Structure

```json
{
  "text": "I'm extremely disappointed!",
  "timestamp": "2025-01-20T10:30:00",
  "language": "english",
  "model": "monologg/bert-base-cased-goemotions-original",
  "top_emotion_1": "disappointment",
  "top_emotion_1_score": 0.856,
  "top_emotion_2": "sadness",
  "top_emotion_2_score": 0.742,
  "top_emotion_3": "annoyance",
  "top_emotion_3_score": 0.621,
  "sentiment": "negative",
  "formality": "neutral",
  "formality_score": 0.65,
  "tone": "rough",
  "tone_score": 0.80,
  "intensity": "high",
  "intensity_score": 0.90,
  "communication_style": "assertive",
  "style_score": 0.85
}
```

---

## Model Management

### Using the Universal Model Loader

Load any HuggingFace text-classification model:

```python
from bahar.models import (
    ModelRegistry,
    UniversalModelLoader,
    ModelInspector,
    UniversalAdapter,
    ModelMetadata
)

# Initialize registry
registry = ModelRegistry()

# Load a model
loader = UniversalModelLoader()
model, tokenizer, config = loader.load_model("cardiffnlp/twitter-roberta-base-emotion")

# Inspect capabilities
capabilities = ModelInspector.inspect_model(model, tokenizer, config)
labels = ModelInspector.extract_labels(config)
taxonomy = ModelInspector.detect_taxonomy(labels)

print(f"Task: {capabilities.task_type}")
print(f"Labels: {capabilities.num_labels}")
print(f"Taxonomy: {taxonomy}")

# Create metadata
metadata = ModelMetadata(
    model_id="cardiffnlp/twitter-roberta-base-emotion",
    name="Twitter RoBERTa Emotion",
    description="Emotion detection trained on Twitter data",
    task_type=capabilities.task_type,
    language=["english"],
    num_labels=capabilities.num_labels,
    label_map=labels,
    taxonomy=taxonomy,
    tags=["emotion", "twitter", "roberta"]
)

# Add to registry
registry.add_model(metadata)

# Use the model
adapter = UniversalAdapter(model, tokenizer, metadata)
result = adapter.predict("I'm so excited!", top_k=3)

print(f"Top predictions: {result.top_predictions}")
```

---

## Language Detection

Automatic language detection:

```python
from bahar.utils.language_models import detect_language

# Detect language
lang = detect_language("This is English text")
print(lang)  # 'english'

lang = detect_language("Dit is Nederlandse tekst")
print(lang)  # 'dutch'

lang = detect_language("این متن فارسی است")
print(lang)  # 'persian'

# Use with analyzer
from bahar import EmotionAnalyzer

analyzer = EmotionAnalyzer(auto_detect_language=True)
analyzer.load_model()

# Automatically detects language and uses appropriate model
result = analyzer.analyze("Your text in any language")
```

---

## Available Models

Get list of available models:

```python
from bahar.utils.language_models import (
    get_available_models,
    get_supported_languages
)

# Get all supported languages
languages = get_supported_languages()
print(languages)  # ['english', 'dutch', 'persian']

# Get models for a specific language
models = get_available_models("english")
for key, name in models.items():
    print(f"{key}: {name}")

# Include registry models
models = get_available_models("english", include_registry=True)
```

---

## Error Handling

Handle errors gracefully:

```python
from bahar import EmotionAnalyzer

try:
    analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
    analyzer.load_model()
    result = analyzer.analyze("Your text")

except RuntimeError as e:
    print(f"Model loading failed: {e}")
    # Handle model loading errors

except ValueError as e:
    print(f"Invalid input: {e}")
    # Handle invalid parameters

except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

---

## Performance Tips

### 1. Reuse Analyzers

Load the model once and reuse for multiple texts:

```python
# Good: Load once
analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
analyzer.load_model()

for text in texts:
    result = analyzer.analyze(text)
    # Process result

# Bad: Load every time
for text in texts:
    analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
    analyzer.load_model()  # Slow!
    result = analyzer.analyze(text)
```

### 2. Use Batch Processing

```python
# Good: Batch processing
results = analyzer.analyze_batch(texts, top_k=3)

# Less efficient: Loop
results = [analyzer.analyze(text, top_k=3) for text in texts]
```

### 3. Choose Appropriate Models

- **Speed**: Smaller models (DistilBERT) are faster
- **Accuracy**: Larger models (RoBERTa) are more accurate
- **Balance**: Base models offer good speed/accuracy trade-off

---

## Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
from bahar import EmotionAnalyzer

app = Flask(__name__)

# Load analyzer once at startup
analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
analyzer.load_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    top_k = data.get('top_k', 3)

    result = analyzer.analyze(text, top_k=top_k)

    return jsonify({
        'emotions': dict(result.get_top_emotions()),
        'sentiment': result.get_sentiment_group()
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Pandas DataFrame

```python
import pandas as pd
from bahar import EmotionAnalyzer

# Load data
df = pd.read_csv('texts.csv')

# Initialize analyzer
analyzer = EmotionAnalyzer(language="english", model_key="goemotions")
analyzer.load_model()

# Analyze all texts
results = analyzer.analyze_batch(df['text'].tolist(), top_k=1)

# Add results to dataframe
df['top_emotion'] = [r.get_top_emotions()[0][0] for r in results]
df['emotion_score'] = [r.get_top_emotions()[0][1] for r in results]
df['sentiment'] = [r.get_sentiment_group() for r in results]

# Save results
df.to_csv('analyzed_texts.csv', index=False)
```

---

## Advanced Usage

### Custom Model Configuration

```python
from bahar import EmotionAnalyzer

# Use explicit model name
analyzer = EmotionAnalyzer(
    model_name="monologg/bert-base-cased-goemotions-ekman"
)
analyzer.load_model()

# Get model info
info = analyzer.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Dataset: {info['dataset']}")
```

### Linguistic Analysis Only

```python
from bahar.analyzers.linguistic_analyzer import LinguisticAnalyzer

analyzer = LinguisticAnalyzer()
features = analyzer.analyze("I am extremely disappointed with this service.")

print(f"Formality: {features.formality}")
print(f"Tone: {features.tone}")
print(f"Intensity: {features.intensity}")
print(f"Style: {features.communication_style}")
```

---

## Complete API Reference

For detailed API documentation, see:

- [Core API Reference](../api/core-api.md)
- [Model Management API](../api/model-management.md)
- [Data Structures](../api/data-structures.md)

---

## Support

- **Documentation**: [Full Documentation](../README.md)
- **Examples**: [Code Examples](../../examples/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

