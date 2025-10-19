# GoEmotions Usage Guide

Comprehensive guide to using GoEmotions in Bahar.

## Basic Usage

### Initialize Analyzer

```python
from bahar import EmotionAnalyzer

# Initialize with GoEmotions dataset
analyzer = EmotionAnalyzer(dataset="goemotions")

# Load the model (downloads ~400MB on first run)
analyzer.load_model()
```

### Analyze Single Text

```python
# Analyze text
result = analyzer.analyze("I'm so excited about this opportunity!", top_k=3)

# Get top emotions
top_emotions = result.get_top_emotions()
for emotion, score in top_emotions:
    print(f"{emotion}: {score:.3f}")

# Get sentiment group
sentiment = result.get_sentiment_group()
print(f"Sentiment: {sentiment}")  # positive, negative, ambiguous, or neutral
```

### Format Output

```python
from bahar.datasets.goemotions.result import format_emotion_output

# Pretty print results
result = analyzer.analyze("Thank you so much for your help!", top_k=3)
print(format_emotion_output(result))
```

Output:
```
Text: Thank you so much for your help!
Sentiment Group: POSITIVE

Top Emotions:
  gratitude       ██████████████████████████████████████████████████ 0.850
  caring          ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░ 0.542
  approval        ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.387
```

## Batch Processing

```python
# Analyze multiple texts
texts = [
    "I'm so happy!",
    "This is terrible.",
    "I'm confused.",
]

results = analyzer.analyze_batch(texts, top_k=3)

for i, result in enumerate(results, 1):
    print(f"\nText {i}: {result.text}")
    print(f"Top emotion: {result.get_top_emotions()[0][0]}")
```

## Multilingual Support

```python
# English
result_en = analyzer.analyze("I'm so grateful for your help!", top_k=3)

# Dutch
result_nl = analyzer.analyze("Ik ben zo blij met dit nieuws!", top_k=3)

# Persian
result_fa = analyzer.analyze("من خیلی خوشحالم!", top_k=3)

# All work with the same analyzer
```

## Access Taxonomy

```python
from bahar.datasets.goemotions import (
    GOEMOTIONS_EMOTIONS,
    EMOTION_GROUPS,
)

# All 28 emotions
print(f"Total emotions: {len(GOEMOTIONS_EMOTIONS)}")
print(GOEMOTIONS_EMOTIONS)

# Emotions by group
for group, emotions in EMOTION_GROUPS.items():
    print(f"\n{group.upper()} ({len(emotions)}):")
    print(f"  {', '.join(emotions)}")
```

## Working with Results

### EmotionResult Object

```python
result = analyzer.analyze("I'm so excited!", top_k=5)

# Access properties
print(result.text)                    # Original text
print(result.emotions)                # Dict of all emotions and scores
print(result.top_k)                   # Number of top emotions

# Get top emotions
top_emotions = result.get_top_emotions()  # List of (emotion, score) tuples

# Get sentiment group
sentiment = result.get_sentiment_group()  # 'positive', 'negative', 'ambiguous', 'neutral'

# Access specific emotion score
joy_score = result.emotions['joy']
print(f"Joy score: {joy_score:.3f}")
```

### Iterate Over All Emotions

```python
result = analyzer.analyze("Your text here")

# Sort by score
sorted_emotions = sorted(
    result.emotions.items(),
    key=lambda x: x[1],
    reverse=True
)

# Show all emotions
for emotion, score in sorted_emotions:
    print(f"{emotion:15s}: {score:.4f}")
```

## Sample Texts

```python
from bahar.datasets.goemotions import SAMPLE_TEXTS
from bahar.datasets.goemotions.samples import get_samples_by_language

# Get all samples
english_samples = SAMPLE_TEXTS["english"]
dutch_samples = SAMPLE_TEXTS["dutch"]
persian_samples = SAMPLE_TEXTS["persian"]

# Or use helper function
samples = get_samples_by_language("english")

# Each sample has:
for sample in samples:
    print(sample["text"])                    # The text
    print(sample["expected_emotion"])        # Expected emotion
    if "translation" in sample:
        print(sample["translation"])         # Translation (non-English)
```

## Custom Model

```python
# Use a different HuggingFace model
analyzer = EmotionAnalyzer(
    dataset="goemotions",
    model_name="your-model-name"
)
analyzer.load_model()
```

## Direct Classifier Access

```python
from bahar.datasets.goemotions import GoEmotionsClassifier

# Use the classifier directly
classifier = GoEmotionsClassifier(
    model_name="monologg/bert-base-cased-goemotions-original"
)
classifier.load_model()

# Same interface
result = classifier.predict("Your text", top_k=3)
```

## Performance Tips

### 1. Reuse Analyzer

```python
# Good: Load model once, reuse analyzer
analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

for text in many_texts:
    result = analyzer.analyze(text)
    # Process result
```

```python
# Bad: Loading model repeatedly
for text in many_texts:
    analyzer = EmotionAnalyzer(dataset="goemotions")
    analyzer.load_model()  # Slow!
    result = analyzer.analyze(text)
```

### 2. Use Batch Processing

```python
# Better for multiple texts
texts = ["text1", "text2", "text3", ...]
results = analyzer.analyze_batch(texts, top_k=3)
```

### 3. Adjust top_k

```python
# Only need top emotion? Use top_k=1
result = analyzer.analyze(text, top_k=1)

# Need all emotions? Use top_k=28
result = analyzer.analyze(text, top_k=28)
```

## Error Handling

```python
try:
    analyzer = EmotionAnalyzer(dataset="goemotions")
    analyzer.load_model()
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("Make sure transformers and torch are installed:")
    print("  uv pip install transformers torch")
```

## Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
from bahar import EmotionAnalyzer

app = Flask(__name__)

# Load model once at startup
analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')

    result = analyzer.analyze(text, top_k=3)

    return jsonify({
        'text': result.text,
        'sentiment': result.get_sentiment_group(),
        'emotions': [
            {'emotion': e, 'score': float(s)}
            for e, s in result.get_top_emotions()
        ]
    })

if __name__ == '__main__':
    app.run()
```

### Pandas DataFrame

```python
import pandas as pd
from bahar import EmotionAnalyzer

analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

# Analyze DataFrame column
df = pd.DataFrame({'text': [
    "I'm so happy!",
    "This is terrible.",
    "I'm confused.",
]})

# Add emotion columns
results = analyzer.analyze_batch(df['text'].tolist(), top_k=1)

df['emotion'] = [r.get_top_emotions()[0][0] for r in results]
df['emotion_score'] = [r.get_top_emotions()[0][1] for r in results]
df['sentiment'] = [r.get_sentiment_group() for r in results]

print(df)
```

### CSV Processing

```python
import csv
from bahar import EmotionAnalyzer

analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

# Read CSV
with open('input.csv', 'r') as f:
    reader = csv.DictReader(f)
    texts = [row['text'] for row in reader]

# Analyze
results = analyzer.analyze_batch(texts, top_k=3)

# Write results
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'top_emotion', 'score', 'sentiment'])

    for result in results:
        top_emotion, score = result.get_top_emotions()[0]
        writer.writerow([
            result.text,
            top_emotion,
            f"{score:.3f}",
            result.get_sentiment_group()
        ])
```

## Advanced Usage

### Filter by Sentiment

```python
results = analyzer.analyze_batch(texts, top_k=1)

# Filter positive texts
positive_texts = [
    r.text for r in results
    if r.get_sentiment_group() == 'positive'
]

# Filter by specific emotion
grateful_texts = [
    r.text for r in results
    if r.get_top_emotions()[0][0] == 'gratitude'
]
```

### Emotion Statistics

```python
from collections import Counter

results = analyzer.analyze_batch(texts, top_k=1)

# Count emotions
emotion_counts = Counter(
    r.get_top_emotions()[0][0] for r in results
)

print("Most common emotions:")
for emotion, count in emotion_counts.most_common(5):
    print(f"  {emotion}: {count}")
```

## See Also

- [Taxonomy](taxonomy.md) - Detailed emotion descriptions
- [API Reference](../api/analyzers.md) - Complete API documentation
- [Linguistic Analysis](../guides/linguistic-analysis.md) - Add linguistic features

