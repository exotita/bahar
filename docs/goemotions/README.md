# GoEmotions Dataset Documentation

## Overview

GoEmotions is a dataset of 58k Reddit comments labeled with 27 emotion categories plus neutral, created by Google Research for fine-grained emotion classification.

**Reference:** [GoEmotions Research Blog](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)

## Key Features

- **28 Emotion Categories**: 27 emotions + neutral
- **Fine-Grained**: More nuanced than basic 6 emotions (joy, sadness, anger, fear, surprise, disgust)
- **Balanced Taxonomy**: 12 positive, 11 negative, 4 ambiguous, 1 neutral
- **Multilingual Support**: Works with English, Dutch, Persian, and more via multilingual models

## Emotion Categories

### Positive Emotions (12)
- admiration
- amusement
- approval
- caring
- desire
- excitement
- gratitude
- joy
- love
- optimism
- pride
- relief

### Negative Emotions (11)
- anger
- annoyance
- disappointment
- disapproval
- disgust
- embarrassment
- fear
- grief
- nervousness
- remorse
- sadness

### Ambiguous Emotions (4)
- confusion
- curiosity
- realization
- surprise

### Neutral (1)
- neutral

## Why GoEmotions?

### 1. **Rich Taxonomy**
Unlike the basic 6 emotions, GoEmotions includes 12 positive emotions, allowing for subtle differentiation in positive sentiment.

### 2. **Conversational Data**
Based on Reddit comments, making it suitable for real-world conversational applications.

### 3. **Academic Validation**
Published by Google Research with rigorous methodology and high inter-rater agreement (94%).

### 4. **Practical Applications**
- Empathetic chatbots
- Content moderation
- Customer support analysis
- Social media monitoring

## Usage in Bahar

### Basic Emotion Analysis

```python
from bahar import EmotionAnalyzer
from bahar.datasets.goemotions.result import format_emotion_output

# Initialize analyzer
analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()

# Analyze text
result = analyzer.analyze("I'm so excited about this!", top_k=3)
print(format_emotion_output(result))
```

### Access Taxonomy

```python
from bahar.datasets.goemotions import GOEMOTIONS_EMOTIONS, EMOTION_GROUPS

# All 28 emotions
print(f"Total emotions: {len(GOEMOTIONS_EMOTIONS)}")

# Grouped by sentiment
for group, emotions in EMOTION_GROUPS.items():
    print(f"{group}: {emotions}")
```

### Sample Texts

```python
from bahar.datasets.goemotions import SAMPLE_TEXTS

# Get samples by language
english_samples = SAMPLE_TEXTS["english"]
dutch_samples = SAMPLE_TEXTS["dutch"]
persian_samples = SAMPLE_TEXTS["persian"]
```

## Model Information

**Default Model:** `monologg/bert-base-cased-goemotions-original`
- Pre-trained on GoEmotions dataset
- ~400MB download on first use
- Optimized for English text
- Works with other languages via multilingual embeddings

**Custom Models:**
```python
# Use a different model
analyzer = EmotionAnalyzer(
    dataset="goemotions",
    model_name="your-model-name"
)
```

## Performance

- **Accuracy**: Based on pre-trained model performance
- **Speed**: ~50-100ms per text (model-dependent)
- **Languages**: Best for English, good for Dutch/Persian
- **Max Length**: 512 tokens

## Limitations

1. **Language Bias**: Optimized for English
2. **Context Dependency**: Emotions are subtle and context-dependent
3. **Sarcasm**: May not detect sarcasm or irony
4. **Cultural Differences**: Emotion expression varies by culture

## Research Paper

**Title:** "GoEmotions: A Dataset of Fine-Grained Emotions"

**Authors:** Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, Sujith Ravi

**Published:** 2020

**Citation:**
```bibtex
@inproceedings{demszky2020goemotions,
  title={GoEmotions: A Dataset of Fine-Grained Emotions},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```

## Additional Resources

- [Taxonomy Details](taxonomy.md) - Detailed emotion descriptions
- [Usage Examples](usage.md) - Code examples and tutorials
- [Sample Texts](samples.md) - Multilingual sample documentation

## Related Datasets

For comparison with other emotion datasets:
- EmoBank (Valence-Arousal-Dominance)
- ISEAR (International Survey on Emotion Antecedents and Reactions)
- SemEval-2018 Task 1 (Affect in Tweets)

See [Adding New Datasets](../guides/adding-datasets.md) for how to integrate other datasets.

