#!/bin/bash
# Bahar Streamlit App Launcher

echo "🌸 Starting Bahar Web Application..."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating .venv..."
    source .venv/bin/activate
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing streamlit..."
    uv pip install streamlit
fi

# Run the app
echo "🚀 Launching Streamlit app..."
echo "📍 URL: http://localhost:8501"
echo ""
streamlit run app.py

