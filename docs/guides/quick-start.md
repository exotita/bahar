# Quick Start Guide

## Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Dependencies already installed:
# - transformers>=4.57.0
# - torch>=2.9.0
```

## Run Demos

### Basic Demo (Emotions Only)
```bash
python main.py
```

### Enhanced Demo (Emotions + Linguistics)
```bash
python demo_enhanced.py
```

### Test All Linguistic Categories
```bash
python test_linguistic_categories.py
```

This will test 16 linguistic categories with 48 multilingual samples.

## Classify Your Own Text

```bash
# English
python classify_text.py "I'm so excited about this opportunity!"

# Dutch
python classify_text.py "Ik ben zo blij met dit nieuws!"

# Persian
python classify_text.py "من از این خبر خیلی خوشحالم!"

# Show top 5 emotions instead of 3
python classify_text.py "Your text here" --top-k 5
```

## Use in Python Code

```python
from emotion_classifier import MultilingualEmotionClassifier, format_emotion_output

# Initialize
classifier = MultilingualEmotionClassifier()
classifier.load_model()

# Classify
result = classifier.predict("I love this!", top_k=3)

# Display formatted output
print(format_emotion_output(result))

# Or access raw data
for emotion, score in result.get_top_emotions():
    print(f"{emotion}: {score:.3f}")

sentiment = result.get_sentiment_group()  # positive/negative/ambiguous/neutral
```

## GoEmotions Taxonomy (28 emotions)

**Positive (12):** admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief

**Negative (11):** anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness

**Ambiguous (4):** confusion, curiosity, realization, surprise

**Neutral (1):** neutral

## Notes

- First run downloads ~400MB model from HuggingFace
- Model is optimized for English but works with other languages
- Results are probabilistic; emotions are subtle and context-dependent
- Maximum input length: 512 tokens

## Files

- `main.py` - Demo with sample texts
- `classify_text.py` - CLI for custom text
- `emotion_classifier.py` - Core implementation
- `sample_texts.py` - Sample texts in 3 languages
- `README.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical details

## Help

For detailed information, see `README.md`

