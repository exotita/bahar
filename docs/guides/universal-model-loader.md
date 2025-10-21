# Universal Model Loader - Design Document

**Version:** 1.0
**Date:** 2025-01-20
**Status:** Design Phase

## Executive Summary

This document outlines the design and implementation plan for a Universal Model Loader system that enables dynamic loading and integration of any HuggingFace model into the Bahar emotion analysis system without code changes.

## Current Architecture

### Components
- **GoEmotionsClassifier**: Hardcoded model loading
- **language_models.py**: Static model registry (9 models)
- **model_adapters.py**: Manual adapters for specific model types
- **EmotionResult**: Fixed 28-emotion taxonomy
- **Streamlit UI**: Dropdown from predefined models

### Limitations
1. Cannot add new models without code changes
2. Hardcoded model types and adapters
3. No automatic model capability detection
4. Fixed emotion taxonomy (28 emotions)
5. Manual adapter creation for each model type
6. No model metadata or versioning
7. Limited to sequence classification models

## Proposed Architecture

### Layer 1: Model Registry

**Purpose**: Store and manage model configurations

#### Components

**1.1 ModelMetadata** (`bahar/models/metadata.py`)
```python
@dataclass
class ModelMetadata:
    model_id: str  # HuggingFace model ID
    name: str  # Display name
    description: str
    task_type: str  # "text-classification", "token-classification", etc.
    language: str | list[str]
    num_labels: int
    label_map: dict[int, str]
    taxonomy: str  # "goemotions", "custom", "sentiment", etc.
    added_date: datetime
    last_used: datetime
    use_count: int
    tags: list[str]
    custom_config: dict[str, Any]
```

**1.2 ModelRegistry** (`bahar/models/registry.py`)
```python
class ModelRegistry:
    def __init__(self, storage_path: Path = Path("config/models.json"))
    def add_model(self, metadata: ModelMetadata) -> None
    def remove_model(self, model_id: str) -> None
    def get_model(self, model_id: str) -> ModelMetadata | None
    def list_models(self, filters: dict | None = None) -> list[ModelMetadata]
    def search_models(self, query: str) -> list[ModelMetadata]
    def update_usage(self, model_id: str) -> None
    def save(self) -> None
    def load(self) -> None
```

### Layer 2: Model Loader

**Purpose**: Load any HuggingFace model dynamically

#### Components

**2.1 UniversalModelLoader** (`bahar/models/loader.py`)
```python
class UniversalModelLoader:
    def __init__(self, cache_dir: Path | None = None)
    def load_model(self, model_id: str, **kwargs) -> tuple[Any, Any, dict]
        # Returns: (model, tokenizer, config)
    def load_from_metadata(self, metadata: ModelMetadata) -> tuple[Any, Any, dict]
    def validate_model(self, model_id: str) -> bool
    def get_model_info(self, model_id: str) -> dict
```

**2.2 TokenizerLoader** (`bahar/models/tokenizer_loader.py`)
```python
class TokenizerLoader:
    @staticmethod
    def load_robust(model_id: str, **kwargs) -> Any
        # Multi-strategy tokenizer loading (already implemented)
```

### Layer 3: Model Detection & Inspection

**Purpose**: Automatically detect model capabilities

#### Components

**3.1 ModelInspector** (`bahar/models/inspector.py`)
```python
class ModelInspector:
    @staticmethod
    def inspect_model(model, tokenizer, config) -> ModelCapabilities
    @staticmethod
    def detect_task_type(config) -> str
    @staticmethod
    def extract_labels(config) -> dict[int, str]
    @staticmethod
    def detect_taxonomy(labels: dict) -> str
    @staticmethod
    def get_supported_languages(model_id: str) -> list[str]
```

**3.2 ModelCapabilities** (`bahar/models/capabilities.py`)
```python
@dataclass
class ModelCapabilities:
    task_type: str
    supports_batch: bool
    max_length: int
    num_labels: int
    label_type: str  # "emotion", "sentiment", "custom"
    output_format: str  # "logits", "probabilities", "scores"
    special_features: list[str]
```

### Layer 4: Universal Adapter

**Purpose**: Unified interface for all models

#### Components

**4.1 UniversalAdapter** (`bahar/models/adapter.py`)
```python
class UniversalAdapter:
    def __init__(self, model, tokenizer, metadata: ModelMetadata)
    def predict(self, text: str, top_k: int = 3) -> UniversalResult
    def predict_batch(self, texts: list[str], top_k: int = 3) -> list[UniversalResult]
    def _normalize_output(self, raw_output) -> dict[str, float]
```

**4.2 UniversalResult** (`bahar/models/result.py`)
```python
@dataclass
class UniversalResult:
    text: str
    model_id: str
    task_type: str
    predictions: dict[str, float]  # label -> score
    top_predictions: list[tuple[str, float]]
    metadata: dict[str, Any]
    raw_output: Any  # Original model output

    def to_emotion_result(self) -> EmotionResult
        # Convert to legacy format
    def to_dict(self) -> dict
    def to_json(self) -> str
```

**4.3 ResultNormalizer** (`bahar/models/normalizer.py`)
```python
class ResultNormalizer:
    @staticmethod
    def normalize_sentiment(predictions: dict) -> dict[str, float]
    @staticmethod
    def normalize_emotions(predictions: dict, taxonomy: str) -> dict[str, float]
    @staticmethod
    def map_to_goemotions(predictions: dict) -> dict[str, float]
```

### Layer 5: UI Integration

**Purpose**: User interface for model management

#### Components

**5.1 ModelManager** (Streamlit tab in `app.py`)
- Add new model by HuggingFace ID
- Test model with sample text
- View model details
- Remove models
- Set default models per language

**5.2 ModelBrowser** (Optional - HuggingFace Hub integration)
- Search HuggingFace Hub
- Filter by task, language, popularity
- Preview model card
- One-click add to registry

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Step 1.1**: Create base structure
```bash
mkdir -p bahar/models
touch bahar/models/__init__.py
touch bahar/models/metadata.py
touch bahar/models/registry.py
touch bahar/models/capabilities.py
```

**Step 1.2**: Implement ModelMetadata and ModelRegistry
- Define data structures
- Implement JSON persistence
- Add CRUD operations
- Write unit tests

**Step 1.3**: Migrate existing models to registry
- Convert language_models.py to registry format
- Create migration script
- Update existing code to use registry

### Phase 2: Model Loading (Week 1-2)

**Step 2.1**: Implement UniversalModelLoader
- Dynamic model loading from HuggingFace
- Integration with TokenizerLoader
- Error handling and validation
- Caching support

**Step 2.2**: Implement ModelInspector
- Auto-detect task type
- Extract label mappings
- Detect taxonomy
- Language detection

**Step 2.3**: Test with various model types
- Text classification (emotions, sentiment)
- Multi-label classification
- Different architectures (BERT, RoBERTa, ALBERT, etc.)

### Phase 3: Universal Adapter (Week 2)

**Step 3.1**: Implement UniversalResult
- Flexible result structure
- Conversion to legacy formats
- JSON serialization

**Step 3.2**: Implement UniversalAdapter
- Unified prediction interface
- Output normalization
- Batch processing

**Step 3.3**: Implement ResultNormalizer
- Sentiment mapping
- Emotion mapping
- Custom taxonomy support

### Phase 4: Integration (Week 2-3)

**Step 4.1**: Update EmotionAnalyzer
- Use UniversalModelLoader
- Support dynamic models
- Maintain backward compatibility

**Step 4.2**: Update EnhancedAnalyzer
- Integrate with new system
- Support multiple model types

**Step 4.3**: Update CLI tools
- Support model registry
- Add model management commands

### Phase 5: UI Development (Week 3)

**Step 5.1**: Add Model Management tab to Streamlit
- Model list view
- Add model form
- Model details view
- Test model interface

**Step 5.2**: Update Analysis tab
- Dynamic model selection from registry
- Show model metadata
- Display model-specific results

**Step 5.3**: Add Model Browser (Optional)
- HuggingFace Hub search
- Model filtering
- Preview and add

### Phase 6: Testing & Documentation (Week 3-4)

**Step 6.1**: Comprehensive testing
- Unit tests for all components
- Integration tests
- Test with 20+ different models
- Performance testing

**Step 6.2**: Documentation
- API documentation
- User guide
- Model addition guide
- Migration guide

**Step 6.3**: Examples and demos
- Jupyter notebook examples
- CLI examples
- Various model types

## Technical Specifications

### Model Storage Format

**config/models.json**:
```json
{
  "models": [
    {
      "model_id": "monologg/bert-base-cased-goemotions-original",
      "name": "GoEmotions BERT",
      "description": "BERT model fine-tuned on GoEmotions dataset",
      "task_type": "text-classification",
      "language": ["english"],
      "num_labels": 28,
      "label_map": {"0": "admiration", "1": "amusement", ...},
      "taxonomy": "goemotions",
      "added_date": "2025-01-20T10:00:00Z",
      "last_used": "2025-01-20T15:30:00Z",
      "use_count": 42,
      "tags": ["emotion", "english", "bert"],
      "custom_config": {}
    }
  ],
  "version": "1.0"
}
```

### API Examples

**Adding a new model**:
```python
from bahar.models import ModelRegistry, UniversalModelLoader, ModelInspector

# Initialize
registry = ModelRegistry()
loader = UniversalModelLoader()

# Load and inspect model
model, tokenizer, config = loader.load_model("cardiffnlp/twitter-roberta-base-emotion")
capabilities = ModelInspector.inspect_model(model, tokenizer, config)

# Create metadata
metadata = ModelMetadata(
    model_id="cardiffnlp/twitter-roberta-base-emotion",
    name="Twitter RoBERTa Emotion",
    description="RoBERTa model trained on Twitter data",
    task_type=capabilities.task_type,
    language=["english"],
    num_labels=capabilities.num_labels,
    label_map=ModelInspector.extract_labels(config),
    taxonomy="custom",
    added_date=datetime.now(),
    tags=["emotion", "twitter", "roberta"]
)

# Add to registry
registry.add_model(metadata)
registry.save()
```

**Using a model**:
```python
from bahar.models import UniversalAdapter

# Get model from registry
metadata = registry.get_model("cardiffnlp/twitter-roberta-base-emotion")

# Load model
model, tokenizer, config = loader.load_from_metadata(metadata)

# Create adapter
adapter = UniversalAdapter(model, tokenizer, metadata)

# Analyze text
result = adapter.predict("I'm so happy today!", top_k=3)

# Access results
print(result.top_predictions)  # [("joy", 0.95), ("optimism", 0.78), ...]
print(result.to_dict())  # Full result as dict
emotion_result = result.to_emotion_result()  # Convert to legacy format
```

## Backward Compatibility

### Strategy
1. Keep existing EmotionAnalyzer interface
2. Add `use_universal_loader=True` parameter
3. Maintain EmotionResult format
4. Gradual migration path

### Migration Example
```python
# Old way (still works)
analyzer = EmotionAnalyzer(language="english", model_key="goemotions")

# New way (using universal loader)
analyzer = EmotionAnalyzer(
    model_id="monologg/bert-base-cased-goemotions-original",
    use_universal_loader=True
)

# Or from registry
analyzer = EmotionAnalyzer.from_registry("my-custom-model")
```

## Benefits

1. **Flexibility**: Add any HuggingFace model without code changes
2. **Extensibility**: Support new tasks and model types easily
3. **User-Friendly**: UI for model management
4. **Maintainability**: Centralized model configuration
5. **Scalability**: Easy to add hundreds of models
6. **Research-Ready**: Quick experimentation with different models
7. **Production-Ready**: Robust error handling and validation

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model incompatibility | High | Comprehensive validation and testing |
| Performance degradation | Medium | Caching and lazy loading |
| Breaking changes | High | Maintain backward compatibility |
| Complex UI | Medium | Iterative design with user feedback |
| Storage overhead | Low | Efficient JSON storage, optional SQLite |

## Success Criteria

1. ✅ Add any HuggingFace text-classification model via UI
2. ✅ Automatic model capability detection (>90% accuracy)
3. ✅ Backward compatibility with existing code
4. ✅ <5% performance overhead vs. current system
5. ✅ Successfully tested with 20+ different models
6. ✅ Complete documentation and examples
7. ✅ User can add and test a new model in <2 minutes

## Timeline

- **Week 1**: Core infrastructure + Model loading
- **Week 2**: Universal adapter + Integration
- **Week 3**: UI development + Testing
- **Week 4**: Documentation + Polish

**Total**: 3-4 weeks for full implementation

## Next Steps

1. Review and approve design
2. Create GitHub issues/tasks
3. Set up development branch
4. Begin Phase 1 implementation
5. Regular progress reviews

---

**Document Status**: Draft
**Last Updated**: 2025-01-20
**Author**: Bahar Development Team

