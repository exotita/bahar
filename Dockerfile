# Bahar - Multilingual Emotion & Linguistic Analysis
# Docker image for production deployment

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/cache/transformers \
    HF_HOME=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    NLTK_DATA=/app/cache/nltk_data \
    SPACY_DATA=/app/cache/spacy_data

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first
COPY pyproject.toml .python-version ./

# Install uv and Python dependencies in one layer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:${PATH}" && \
    uv pip install --system --no-cache \
        transformers>=4.57.0 \
        torch>=2.9.0 \
        rich>=14.2.0 \
        streamlit>=1.31.0 \
        spacy>=3.7.0 \
        pandas>=2.0.0 \
        nltk>=3.9.0 \
        gensim>=4.3.0 \
        scikit-learn>=1.5.0 \
        umap-learn>=0.5.0 \
        matplotlib>=3.9.0 \
        seaborn>=0.13.0 \
        networkx>=3.3 \
        python-Levenshtein>=0.25.0 \
        textacy>=0.13.0 \
        pyphen>=0.16.0 \
        pillow>=10.0.0

# Add uv to PATH for subsequent commands
ENV PATH="/root/.local/bin:${PATH}"

# Download spaCy models
RUN python -m spacy download en_core_web_lg && \
    python -m spacy download nl_core_news_lg && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download nl_core_news_sm

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('wordnet', download_dir='/app/cache/nltk_data'); \
    nltk.download('omw-1.4', download_dir='/app/cache/nltk_data'); \
    nltk.download('averaged_perceptron_tagger', download_dir='/app/cache/nltk_data'); \
    nltk.download('punkt', download_dir='/app/cache/nltk_data'); \
    nltk.download('stopwords', download_dir='/app/cache/nltk_data'); \
    nltk.download('brown', download_dir='/app/cache/nltk_data'); \
    nltk.download('wordnet_ic', download_dir='/app/cache/nltk_data')"

# Copy application code
COPY bahar/ ./bahar/
COPY docs/ ./docs/
COPY config/ ./config/
COPY app.py main.py classify_text.py classify_enhanced.py ./
COPY README.md CHANGELOG.md ./

# Create necessary directories
RUN mkdir -p /app/cache/transformers \
    /app/cache/huggingface \
    /app/cache/torch \
    /app/cache/nltk_data \
    /app/cache/spacy_data \
    /app/config \
    /app/data

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true", \
     "--browser.gatherUsageStats=false"]

