# Installation Guide

Complete installation instructions for Bahar.

## Requirements

- Python 3.12 or higher
- pip or uv (recommended)
- Virtual environment (recommended)

## Quick Installation

### 1. Clone or Download

```bash
cd /path/to/your/projects
# If using git
git clone <repository-url> bahar
cd bahar
```

### 2. Check Python Version

```bash
python --version  # Should be 3.12 or higher
```

The project includes a `.python-version` file that specifies Python 3.12.

### 3. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv

# Or using standard venv
python -m venv .venv
```

### 4. Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```cmd
.venv\Scripts\activate
```

### 5. Install Dependencies

**Using uv (recommended):**
```bash
uv pip install transformers torch
```

**Or sync from pyproject.toml:**
```bash
uv sync
```

**Using pip:**
```bash
pip install transformers torch
```

## Verify Installation

```bash
# Test imports
python -c "from bahar import EmotionAnalyzer; print('✓ Installation successful')"

# Check GoEmotions
python -c "from bahar.datasets.goemotions import GOEMOTIONS_EMOTIONS; print(f'✓ {len(GOEMOTIONS_EMOTIONS)} emotions loaded')"
```

## First Run

The first time you run emotion classification, the model will be downloaded (~400MB):

```bash
python main.py
```

This will:
1. Download the GoEmotions model from HuggingFace
2. Cache it locally
3. Run the demo with sample texts

## Package Structure

After installation, you'll have:

```
bahar/
├── bahar/                  # Main package
│   ├── datasets/           # Dataset modules
│   ├── analyzers/          # Analysis modules
│   ├── cli/                # CLI tools
│   └── demos/              # Demo scripts
├── docs/                   # Documentation
├── main.py                 # Demo script
├── classify_text.py        # Basic CLI
├── classify_enhanced.py    # Enhanced CLI
└── emotion_classification_demo.ipynb  # Jupyter notebook
```

## Optional: Jupyter Notebook

To use the Jupyter notebook:

```bash
# Install Jupyter
uv pip install jupyter

# Or with pip
pip install jupyter

# Start Jupyter
jupyter notebook emotion_classification_demo.ipynb
```

## Troubleshooting

### Python Version Issues

If you have multiple Python versions:

```bash
# Use specific Python version
python3.12 -m venv .venv
source .venv/bin/activate
```

### Import Errors

If you get import errors:

```bash
# Make sure you're in the project directory
cd /path/to/bahar

# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv pip install --force-reinstall transformers torch
```

### Model Download Issues

If model download fails:

1. Check internet connection
2. Try again (downloads can be interrupted)
3. Clear cache and retry:
   ```bash
   rm -rf ~/.cache/huggingface
   python main.py
   ```

### Permission Issues

On macOS/Linux, if you get permission errors:

```bash
# Don't use sudo with pip/uv
# Instead, use virtual environment (recommended above)
```

## Development Installation

For development with additional tools:

```bash
# Install development dependencies
uv pip install transformers torch pytest ruff pyright

# Or with pip
pip install transformers torch pytest ruff pyright
```

## Updating

To update to the latest version:

```bash
# Pull latest changes (if using git)
git pull

# Update dependencies
uv pip install --upgrade transformers torch

# Or with pip
pip install --upgrade transformers torch
```

## Uninstallation

To remove Bahar:

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf bahar
```

## Next Steps

After installation:

1. [Quick Start Guide](quick-start.md) - Get started in 5 minutes
2. [GoEmotions Overview](../goemotions/README.md) - Learn about the dataset
3. [API Documentation](../api/analyzers.md) - Explore the API

## System Requirements

### Minimum
- **RAM**: 4GB
- **Disk**: 1GB free space
- **CPU**: Any modern processor
- **OS**: macOS, Linux, Windows

### Recommended
- **RAM**: 8GB or more
- **Disk**: 2GB free space
- **CPU**: Multi-core processor
- **GPU**: Optional (for faster inference)

## GPU Support

To use GPU acceleration (optional):

```bash
# Install PyTorch with CUDA support
# Visit https://pytorch.org for specific instructions

# Example for CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Docker Installation (Optional)

Coming soon: Docker container for easy deployment.

## Support

If you encounter issues:

1. Check this installation guide
2. Review [Troubleshooting](#troubleshooting) section
3. Check the main [README](../../README.md)
4. Open an issue on GitHub (if applicable)

