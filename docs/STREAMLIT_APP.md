# Bahar Streamlit Web Application

Professional web interface for multilingual emotion and linguistic analysis.

## Features

### 🎯 Analysis Tab
- **Basic Emotion Analysis**: Detect 28 fine-grained emotions
- **Enhanced Analysis**: Emotion + linguistic dimensions
- **Multilingual Support**: English, Dutch, Persian
- **Export Options**: Download results as JSON or CSV
- **Real-time Analysis**: Instant results with progress indicators

### 📚 Samples Tab
- Pre-loaded sample texts for testing
- Organized by language
- Quick analysis with one click
- Category-based organization

### ⚙️ Configuration Tab
- **Taxonomy Management**: Edit emotion list
- **Emotion Groups**: Customize sentiment categories
- **Sample Management**: Add/edit/delete sample texts
- **JSON-based Configuration**: Easy to backup and share
- **Reset Options**: Restore defaults anytime

### 📖 Documentation Tab
- System overview
- Emotion categories reference
- Linguistic dimensions guide
- API usage examples

## Installation

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install streamlit
uv pip install streamlit

# Or install all dependencies
uv sync
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Configuration Files

The app creates a `config/` directory with JSON files:

```
config/
├── taxonomy.json          # List of emotions
├── emotion_groups.json    # Sentiment categories
└── samples.json          # Sample texts by language
```

### Taxonomy Format (`taxonomy.json`)

```json
[
  "admiration",
  "amusement",
  "anger",
  ...
]
```

### Emotion Groups Format (`emotion_groups.json`)

```json
{
  "positive": ["admiration", "amusement", "approval", ...],
  "negative": ["anger", "annoyance", "disappointment", ...],
  "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
  "neutral": ["neutral"]
}
```

### Samples Format (`samples.json`)

```json
{
  "english": [
    {
      "text": "I'm so happy and excited!",
      "category": "positive"
    }
  ],
  "dutch": [...],
  "persian": [...]
}
```

## Usage

### Basic Analysis

1. Go to the **Analysis** tab
2. Select "Basic Emotion"
3. Enter your text
4. Choose language and top-k emotions
5. Click "🔍 Analyze"

### Enhanced Analysis

1. Go to the **Analysis** tab
2. Select "Enhanced (Emotion + Linguistics)"
3. Enter your text
4. Click "🔍 Analyze"
5. View emotion and linguistic results
6. Download results as JSON or CSV

### Testing with Samples

1. Go to the **Samples** tab
2. Select a language
3. Click "Analyze Sample" for any sample
4. View results instantly

### Customizing Configuration

1. Go to the **Configuration** tab
2. Choose what to edit:
   - **Taxonomy**: Add/remove emotions
   - **Emotion Groups**: Reorganize categories
   - **Samples**: Add new test cases
3. Make your changes
4. Click "💾 Save"
5. Reset to defaults anytime with "🔄 Reset"

## UI Features

### Professional Design
- Clean, modern interface
- Color-coded sentiment display
- Progress bars for confidence scores
- Responsive layout
- Professional typography

### Color Scheme
- **Positive**: Green border
- **Negative**: Red border
- **Ambiguous**: Yellow border
- **Neutral**: Gray border

### Metrics Display
- Formality level with progress bar
- Tone analysis with visual indicator
- Intensity measurement
- Communication style classification

## Advanced Features

### Model Caching
The app uses Streamlit's `@st.cache_resource` to cache models:
- Models load once and stay in memory
- Fast subsequent analyses
- Efficient resource usage

### Export Formats

#### JSON Export
```json
{
  "text": "Your text",
  "sentiment_group": "positive",
  "primary_emotion": "joy",
  "primary_emotion_score": 0.8523,
  "formality": "formal",
  "tone": "friendly",
  ...
}
```

#### CSV Export
```csv
field,value
text,Your text
sentiment_group,positive
primary_emotion,joy
primary_emotion_score,0.8523
...
```

## Deployment

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

### Production Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click

#### Docker
```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting

### Model Loading Issues
- Ensure transformers and torch are installed
- Check internet connection (first run downloads model)
- Model size: ~400MB

### Configuration Not Saving
- Check write permissions in `config/` directory
- Ensure valid JSON format
- Check file paths

### Performance Issues
- Models are cached after first load
- Use smaller `top_k` values for faster results
- Close other browser tabs

## Tips

1. **First Run**: Model download takes time (~400MB)
2. **Batch Testing**: Use Samples tab for quick tests
3. **Export Data**: Download results for further analysis
4. **Custom Taxonomy**: Edit to match your use case
5. **Backup Config**: Save `config/` directory regularly

## API Integration

The Streamlit app uses the same API as the CLI:

```python
from bahar import EmotionAnalyzer, EnhancedAnalyzer

# Basic analysis
analyzer = EmotionAnalyzer(dataset="goemotions")
analyzer.load_model()
result = analyzer.analyze("Your text", top_k=3)

# Enhanced analysis
enhanced = EnhancedAnalyzer(emotion_dataset="goemotions")
enhanced.load_model()
result = enhanced.analyze("Your text", top_k=3)
```

## Support

For issues or questions:
- Check the Documentation tab in the app
- Review this guide
- Check the main README.md
- Open an issue on GitHub

---

**Version:** 0.2.0
**License:** MIT
**Author:** Bahar Team

