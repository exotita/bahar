### About baarsh

**baarsh** is a multilingual emotion and linguistic analysis system that combines:

- **GoEmotions Dataset**: 28 fine-grained emotion categories
- **Linguistic Analysis**: Formality, tone, intensity, and communication style
- **Multilingual Support**: English, Dutch, Persian, and more
- **Universal Model Loader**: Load any HuggingFace text-classification model dynamically

### 📖 How to Use

1. **Analysis Tab**: Enter text and choose analysis type (Basic or Enhanced)
2. **Samples Tab**: Test with pre-loaded examples in multiple languages
3. **Model Management Tab**: Add, edit, and test HuggingFace models dynamically
4. **Configuration Tab**: Customize taxonomy, emotion groups, and samples
5. **Documentation Tab**: Learn about the system and API

### ✨ Features

- 🎭 **28 Fine-Grained Emotions** - GoEmotions taxonomy with detailed emotion categories
- 🌍 **Multilingual Support** - English, Dutch, Persian with language-specific models
- 📊 **Linguistic Analysis** - Formality, tone, intensity, and communication style detection
- ⚙️ **Configurable Taxonomy** - Customize emotions, groups, and samples
- 💾 **Export Results** - Download analysis results in JSON/CSV format
- 📝 **Real-Time Analysis** - Instant emotion and linguistic analysis
- 🔧 **Academic Research Ready** - Structured data export for research
- 🤖 **Dynamic Model Management** - Load and test any HuggingFace model
- 🔄 **Auto-Detection** - Automatic language and capability detection
- 📋 **Label Editing** - Customize model labels and mappings
- 🧪 **Model Testing** - Test models before using in production

### 🎯 Use Cases

- **Research**: Academic studies on emotion and language
- **Content Analysis**: Analyze social media, reviews, feedback
- **Sentiment Monitoring**: Track emotional trends in text data
- **Linguistic Studies**: Analyze formality, tone, and style
- **Multilingual Analysis**: Compare emotions across languages
- **Model Evaluation**: Test and compare different emotion models

### 🚀 Getting Started

**Quick Start:**
1. Go to the **Analysis** tab
2. Select your language (English, Dutch, or Persian)
3. Choose a model (or use default)
4. Enter or paste your text
5. Click **🔍 Analyze**

**Try Samples:**
1. Go to the **Samples** tab
2. Select a language
3. Click **🔍 Analyze** on any sample
4. View instant results

**Add Custom Models:**
1. Go to **Model Management** → **Add Model**
2. Enter a HuggingFace model ID
3. System auto-detects capabilities
4. Model becomes available in Analysis tab

### 📊 Analysis Types

**Basic Emotion Analysis:**
- Emotion classification with confidence scores
- Top-K emotion ranking
- Sentiment group classification (positive/negative/ambiguous/neutral)
- Visual confidence bars
- All emotion scores available

**Enhanced Analysis:**
- All basic emotion features
- **Formality**: formal, colloquial, neutral
- **Tone**: friendly, rough, serious, kind, neutral
- **Intensity**: high, medium, low
- **Communication Style**: direct, indirect, assertive, passive
- Export to JSON/CSV for research

### 🌐 Supported Languages

| Language | Code | Models | Description |
|----------|------|--------|-------------|
| **English** | `english` | 3+ models | GoEmotions, RoBERTa, DistilRoBERTa |
| **Dutch** | `dutch` | 3+ models | BERTje, multilingual sentiment |
| **Persian** | `persian` | 3+ models | ParsBERT, ALBERT, sentiment |

### 🎭 Emotion Categories

**28 Fine-Grained Emotions** organized into 4 sentiment groups:

- **Positive** (12): admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
- **Negative** (11): anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
- **Ambiguous** (4): confusion, curiosity, realization, surprise
- **Neutral** (1): neutral

### 📈 Model Performance

Models are optimized for:
- **Accuracy**: High-quality emotion detection
- **Speed**: Fast inference for real-time analysis
- **Multilingual**: Language-specific models for better accuracy
- **Flexibility**: Support for various model architectures

### 🔒 Privacy & Data

- **No Data Storage**: Text is processed in real-time, not stored
- **Local Processing**: All analysis happens on your machine
- **Open Source**: Transparent, auditable code
- **Configurable**: Full control over models and settings

### 📌 Version

**v0.2.0** - Beta Version

**What's New:**
- Universal model loader system
- Dynamic model management
- Label editing capabilities
- Enhanced UI with better visualizations
- Improved multilingual support
- Model registry with usage tracking

### 🔗 Links

- **GitHub**: [Repository](https://github.com/baarsh/baarsh)
- **Documentation**: [Full Docs](../README.md)
- **API Reference**: [API Docs](../api/README.md)
- **Quick Start**: [Quick Start Guide](../guides/quick-start.md)

### 💡 Tips

- **Model Selection**: Different models have different strengths - experiment to find the best for your use case
- **Language Detection**: The system can auto-detect language, but manual selection is more accurate
- **Batch Processing**: Use the Samples tab or Python API for analyzing multiple texts
- **Export Data**: Use Enhanced mode for structured data export suitable for research
- **Custom Models**: Add specialized models from HuggingFace for domain-specific analysis

