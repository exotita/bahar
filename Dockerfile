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

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy dependency files
COPY pyproject.toml .python-version ./

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -r pyproject.toml

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

