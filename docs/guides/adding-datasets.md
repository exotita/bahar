# Adding New Datasets

Guide to extending Bahar with additional emotion datasets.

## Overview

Bahar's modular architecture makes it easy to add new emotion datasets alongside GoEmotions. This guide shows you how to integrate datasets like EmoBank, ISEAR, or your own custom dataset.

## Dataset Structure

Each dataset should be organized in its own directory under `bahar/datasets/`:

```
bahar/datasets/
├── goemotions/          # Existing
│   ├── __init__.py
│   ├── taxonomy.py
│   ├── classifier.py
│   ├── result.py
│   └── samples.py
└── your_dataset/        # New dataset
    ├── __init__.py
    ├── taxonomy.py      # Dataset-specific definitions
    ├── classifier.py    # Classifier implementation
    ├── result.py        # Result classes
    └── samples.py       # Sample texts (optional)
```

## Step-by-Step Guide

### Step 1: Create Directory Structure

```bash
cd bahar/datasets
mkdir your_dataset
touch your_dataset/__init__.py
touch your_dataset/taxonomy.py
touch your_dataset/classifier.py
touch your_dataset/result.py
touch your_dataset/samples.py
```

### Step 2: Define Taxonomy

Create `taxonomy.py` with your dataset's emotion/dimension definitions:

```python
# bahar/datasets/your_dataset/taxonomy.py
"""
Your Dataset taxonomy definitions.
"""

from __future__ import annotations

from typing import Final

# Define your emotions/dimensions
YOUR_DATASET_CATEGORIES: Final[list[str]] = [
    "category1",
    "category2",
    "category3",
    # ...
]

# Optional: Group categories
CATEGORY_GROUPS: Final[dict[str, list[str]]] = {
    "group1": ["category1", "category2"],
    "group2": ["category3"],
}
```

### Step 3: Create Result Class

Create `result.py` for result handling:

```python
# bahar/datasets/your_dataset/result.py
"""Result classes for Your Dataset."""

from __future__ import annotations


class YourDatasetResult:
    """Result of classification using Your Dataset."""

    def __init__(
        self,
        text: str,
        predictions: dict[str, float],
        top_k: int = 3,
    ) -> None:
        self.text: str = text
        self.predictions: dict[str, float] = predictions
        self.top_k: int = top_k

    def get_top_predictions(self) -> list[tuple[str, float]]:
        """Get top-k predictions sorted by score."""
        sorted_preds = sorted(
            self.predictions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_preds[: self.top_k]

    def __repr__(self) -> str:
        top = self.get_top_predictions()
        preds_str = ", ".join([f"{cat}: {score:.3f}" for cat, score in top])
        return f"YourDatasetResult(text='{self.text[:50]}...', top=[{preds_str}])"


def format_output(result: YourDatasetResult) -> str:
    """Format result for display."""
    lines: list[str] = []
    lines.append(f"\nText: {result.text}")
    lines.append("\nTop Predictions:")
    for category, score in result.get_top_predictions():
        bar_length = int(score * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        lines.append(f"  {category:15s} {bar} {score:.3f}")
    return "\n".join(lines)
```

### Step 4: Implement Classifier

Create `classifier.py` with your classification logic:

```python
# bahar/datasets/your_dataset/classifier.py
"""
Your Dataset classifier implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

from bahar.datasets.your_dataset.result import YourDatasetResult
from bahar.datasets.your_dataset.taxonomy import YOUR_DATASET_CATEGORIES


class YourDatasetClassifier:
    """
    Classifier for Your Dataset.
    """

    def __init__(self, model_name: str = "your-model-name") -> None:
        """
        Initialize the classifier.

        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name: str = model_name
        self._model = None
        self._tokenizer = None
        self._label_map: dict[int, str] = {}

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers library not installed. "
                "Install with: uv pip install transformers torch"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        # Create label mapping
        if hasattr(self._model.config, "id2label"):
            self._label_map = self._model.config.id2label
        else:
            self._label_map = {
                i: label for i, label in enumerate(YOUR_DATASET_CATEGORIES)
            }

    def predict(self, text: str, top_k: int = 3) -> YourDatasetResult:
        """
        Predict categories for text.

        Args:
            text: Input text
            top_k: Number of top predictions to return

        Returns:
            YourDatasetResult with predictions
        """
        if self._model is None or self._tokenizer is None:
            self.load_model()

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch library not installed. "
                "Install with: uv pip install torch"
            ) from exc

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            # Adjust based on your model's output
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # Map to labels
        predictions: dict[str, float] = {}
        for idx, prob in enumerate(probs[0].tolist()):
            label = self._label_map.get(idx, f"label_{idx}")
            predictions[label] = prob

        return YourDatasetResult(text=text, predictions=predictions, top_k=top_k)

    def predict_batch(
        self, texts: Sequence[str], top_k: int = 3
    ) -> list[YourDatasetResult]:
        """Predict for multiple texts."""
        return [self.predict(text, top_k=top_k) for text in texts]
```

### Step 5: Create Package Exports

Update `__init__.py`:

```python
# bahar/datasets/your_dataset/__init__.py
"""
Your Dataset integration.
"""

from __future__ import annotations

from bahar.datasets.your_dataset.classifier import YourDatasetClassifier
from bahar.datasets.your_dataset.taxonomy import (
    CATEGORY_GROUPS,
    YOUR_DATASET_CATEGORIES,
)

__all__ = [
    "YourDatasetClassifier",
    "YOUR_DATASET_CATEGORIES",
    "CATEGORY_GROUPS",
]
```

### Step 6: Register in EmotionAnalyzer

Update `bahar/analyzers/emotion_analyzer.py`:

```python
def __init__(
    self,
    dataset: str = "goemotions",
    model_name: str | None = None,
) -> None:
    """Initialize emotion analyzer."""
    self.dataset: str = dataset

    if dataset == "goemotions":
        if model_name is None:
            model_name = "monologg/bert-base-cased-goemotions-original"
        from bahar.datasets.goemotions import GoEmotionsClassifier
        self.classifier = GoEmotionsClassifier(model_name)

    elif dataset == "your_dataset":  # Add this
        if model_name is None:
            model_name = "your-default-model"
        from bahar.datasets.your_dataset import YourDatasetClassifier
        self.classifier = YourDatasetClassifier(model_name)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
```

### Step 7: Use Your Dataset

```python
from bahar import EmotionAnalyzer

# Use your dataset
analyzer = EmotionAnalyzer(dataset="your_dataset")
analyzer.load_model()

result = analyzer.analyze("Your text here", top_k=3)
print(result)
```

## Example: EmoBank Dataset

Here's a complete example for the EmoBank dataset (Valence-Arousal-Dominance):

### taxonomy.py

```python
from typing import Final

EMOBANK_DIMENSIONS: Final[list[str]] = [
    "valence",    # Pleasure/displeasure
    "arousal",    # Activation/deactivation
    "dominance",  # Control/lack of control
]

DIMENSION_RANGES: Final[dict[str, tuple[float, float]]] = {
    "valence": (1.0, 5.0),
    "arousal": (1.0, 5.0),
    "dominance": (1.0, 5.0),
}
```

### result.py

```python
class EmoBankResult:
    def __init__(
        self,
        text: str,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> None:
        self.text = text
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance

    def get_vad_vector(self) -> tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)

    def __repr__(self) -> str:
        return (
            f"EmoBankResult(V={self.valence:.2f}, "
            f"A={self.arousal:.2f}, D={self.dominance:.2f})"
        )
```

## Testing Your Dataset

Create tests to verify your implementation:

```python
# tests/test_your_dataset.py
def test_classifier_loading():
    from bahar.datasets.your_dataset import YourDatasetClassifier

    classifier = YourDatasetClassifier()
    # Test without loading model first
    assert classifier._model is None

def test_prediction():
    from bahar import EmotionAnalyzer

    analyzer = EmotionAnalyzer(dataset="your_dataset")
    analyzer.load_model()

    result = analyzer.analyze("Test text", top_k=3)
    assert result.text == "Test text"
    assert len(result.get_top_predictions()) == 3
```

## Documentation

Create documentation for your dataset:

```
docs/your_dataset/
├── README.md       # Overview
├── taxonomy.md     # Category descriptions
└── usage.md        # Usage examples
```

## Best Practices

1. **Follow Naming Conventions**
   - Use lowercase with underscores for module names
   - Use PascalCase for class names
   - Match GoEmotions structure

2. **Type Annotations**
   - Add type hints to all functions
   - Use `from __future__ import annotations`

3. **Error Handling**
   - Provide clear error messages
   - Handle missing dependencies gracefully

4. **Documentation**
   - Document all public methods
   - Include usage examples
   - Explain dataset-specific concepts

5. **Testing**
   - Write unit tests
   - Test edge cases
   - Verify model loading

## Common Datasets to Add

### EmoBank
- **Type:** Dimensional (VAD)
- **Model:** Regression-based
- **Use case:** Emotion dimensions

### ISEAR
- **Type:** Categorical (7 emotions)
- **Model:** Classification
- **Use case:** Cross-cultural emotions

### SemEval-2018 Task 1
- **Type:** Multi-label emotions
- **Model:** Multi-label classification
- **Use case:** Twitter emotion analysis

## See Also

- [GoEmotions Implementation](../../bahar/datasets/goemotions/) - Reference implementation
- [EmotionAnalyzer API](../api/analyzers.md) - Analyzer interface
- [Project Structure](restructure-summary.md) - Overall architecture

