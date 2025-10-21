# Streamlit App Documentation

This directory contains the documentation displayed in the Bahar Streamlit web application.

## Files

| File | Description | Lines | Used In |
|------|-------------|-------|---------|
| `overview.md` | About Bahar, features, getting started | 138 | Overview tab |
| `emotions.md` | 28 emotion categories, groups, examples | 210 | Emotions tab |
| `linguistic-dimensions.md` | Formality, tone, intensity, style | 450 | Linguistic Dimensions tab |
| `api-usage.md` | Code examples, integration, best practices | 505 | API tab |

**Total:** 1,303 lines of documentation

## Usage

These markdown files are loaded dynamically by the Streamlit app (`app.py`) in the Documentation tab:

```python
# Documentation directory
docs_dir = Path("docs/app")

# Load markdown file
overview_file = docs_dir / "overview.md"
if overview_file.exists():
    with open(overview_file, "r", encoding="utf-8") as f:
        st.markdown(f.read())
```

## Editing

To update the documentation:

1. Edit the relevant `.md` file in this directory
2. Save your changes
3. Refresh the Streamlit app (no restart needed)
4. Changes appear immediately in the Documentation tab

## Structure

### overview.md
- About Bahar
- Features and capabilities
- Use cases
- Getting started guide
- Supported languages
- Version information

### emotions.md
- 28 GoEmotions categories
- Sentiment groups (positive, negative, ambiguous, neutral)
- Detailed emotion descriptions
- Examples for each category
- Use cases by emotion group
- Customization instructions

### linguistic-dimensions.md
- Formality (formal, colloquial, neutral)
- Tone (friendly, rough, serious, kind, neutral)
- Intensity (high, medium, low)
- Communication style (direct, indirect, assertive, passive)
- Examples for each dimension
- Combined analysis examples
- Use cases

### api-usage.md
- Installation instructions
- Quick start examples
- Basic and enhanced analysis
- Multilingual support
- Batch processing
- Export formats
- Model management
- Integration examples (Flask, Pandas)
- Performance tips
- Error handling

## Benefits

- **Maintainability**: Docs in separate files, easier to edit
- **Reusability**: Same docs can be used in multiple places
- **Organization**: Clear separation from code
- **Editability**: Update docs without modifying app code
- **Version Control**: Better diff tracking in git
- **Collaboration**: Multiple people can edit docs simultaneously

## See Also

- [API Documentation](../api/) - Python API reference
- [User Guides](../guides/) - Installation, quick start, etc.
- [GoEmotions Docs](../goemotions/) - Dataset documentation
- [Main Docs](../README.md) - Documentation index

