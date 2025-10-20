#!/usr/bin/env python3
"""
Bahar - Streamlit Web Application

Professional web interface for multilingual emotion and linguistic analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.analyzers.enhanced_analyzer import EnhancedAnalyzer
from bahar.datasets.goemotions.taxonomy import EMOTION_GROUPS, GOEMOTIONS_EMOTIONS
from bahar.utils.language_models import get_available_models


# Page configuration
st.set_page_config(
    page_title="Bahar - Emotion & Linguistic Analysis",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .positive { border-left: 4px solid #28a745; }
    .negative { border-left: 4px solid #dc3545; }
    .ambiguous { border-left: 4px solid #ffc107; }
    .neutral { border-left: 4px solid #6c757d; }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Configuration paths
CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)

TAXONOMY_FILE = CONFIG_DIR / "taxonomy.json"
EMOTION_GROUPS_FILE = CONFIG_DIR / "emotion_groups.json"
SAMPLES_FILE = CONFIG_DIR / "samples.json"


def load_config_file(filepath: Path, default_data: dict | list) -> dict | list:  # type: ignore
    """Load configuration from JSON file or create with defaults."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        save_config_file(filepath, default_data)
        return default_data


def save_config_file(filepath: Path, data: dict | list) -> None:
    """Save configuration to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_taxonomy() -> list[str]:
    """Load emotion taxonomy from JSON."""
    result = load_config_file(TAXONOMY_FILE, list(GOEMOTIONS_EMOTIONS))
    return result if isinstance(result, list) else list(GOEMOTIONS_EMOTIONS)


def load_emotion_groups() -> dict[str, list[str]]:
    """Load emotion groups from JSON."""
    result = load_config_file(EMOTION_GROUPS_FILE, EMOTION_GROUPS)
    return result if isinstance(result, dict) else EMOTION_GROUPS


def load_samples() -> dict[str, list[dict[str, str]]]:
    """Load sample texts from JSON."""
    default_samples: dict[str, list[dict[str, str]]] = {
        "english": [
            {"text": "I'm so happy and excited about this!", "category": "positive"},
            {"text": "This is absolutely terrible and disappointing.", "category": "negative"},
            {"text": "I'm not sure how I feel about this situation.", "category": "ambiguous"},
        ],
        "dutch": [
            {"text": "Ik ben zo blij en enthousiast hierover!", "category": "positive"},
            {"text": "Dit is absoluut verschrikkelijk en teleurstellend.", "category": "negative"},
        ],
        "persian": [
            {"text": "من خیلی خوشحال و هیجان‌زده‌ام!", "category": "positive"},
            {"text": "این واقعاً افتضاح و ناامیدکننده است.", "category": "negative"},
        ],
    }
    result = load_config_file(SAMPLES_FILE, default_samples)
    return result if isinstance(result, dict) else default_samples


@st.cache_resource
def load_emotion_analyzer(language: str, model_key: str) -> EmotionAnalyzer:
    """Load and cache emotion analyzer for specific language and model."""
    analyzer = EmotionAnalyzer(language=language, model_key=model_key)
    analyzer.load_model()
    return analyzer


@st.cache_resource
def load_enhanced_analyzer(language: str, model_key: str) -> EnhancedAnalyzer:
    """Load and cache enhanced analyzer for specific language and model."""
    analyzer = EnhancedAnalyzer(language=language, model_key=model_key)
    analyzer.load_model()
    return analyzer


def display_emotion_result(result, sentiment_colors: dict[str, str]) -> None:
    """Display emotion analysis result."""
    sentiment = result.get_sentiment_group()

    # Sentiment display with color
    sentiment_emoji = {
        "positive": "😊",
        "negative": "😞",
        "ambiguous": "😐",
        "neutral": "😶"
    }

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(
            "Sentiment",
            sentiment.upper(),
            delta=None,
            help="Overall sentiment classification"
        )
    with col2:
        st.markdown(f"### {sentiment_emoji.get(sentiment, '😶')} {sentiment.title()} Sentiment")

    st.divider()

    # Top emotions in columns
    st.subheader("Top Emotions")
    top_emotions = result.get_top_emotions()

    cols = st.columns(len(top_emotions))
    for idx, (emotion, score) in enumerate(top_emotions):
        with cols[idx]:
            st.metric(
                label=emotion.replace("_", " ").title(),
                value=f"{score:.1%}",
                help=f"Confidence: {score:.4f}"
            )
            st.progress(score)


def display_linguistic_features(features) -> None:
    """Display linguistic analysis features."""
    st.subheader("Linguistic Dimensions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📝 Formality")
        st.metric(
            label="Level",
            value=features.formality.title(),
            help=f"Confidence: {features.formality_score:.2%}"
        )
        st.progress(features.formality_score, text=f"{features.formality_score:.1%}")

        st.markdown("##### 💬 Tone")
        st.metric(
            label="Type",
            value=features.tone.title(),
            help=f"Confidence: {features.tone_score:.2%}"
        )
        st.progress(features.tone_score, text=f"{features.tone_score:.1%}")

    with col2:
        st.markdown("##### 🔥 Intensity")
        st.metric(
            label="Level",
            value=features.intensity.title(),
            help=f"Confidence: {features.intensity_score:.2%}"
        )
        st.progress(features.intensity_score, text=f"{features.intensity_score:.1%}")

        st.markdown("##### 🎯 Communication Style")
        st.metric(
            label="Type",
            value=features.communication_style.title(),
            help=f"Confidence: {features.style_score:.2%}"
        )
        st.progress(features.style_score, text=f"{features.style_score:.1%}")


def main() -> None:
    """Main Streamlit application."""
    # Header with logo - fully centered using CSS
    st.markdown(
        """
        <style>
        .center-logo {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    try:
        from PIL import Image
        import base64
        from io import BytesIO

        # Load and encode logo
        logo = Image.open("config/logo.png")
        buffered = BytesIO()
        logo.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Display centered logo with inline CSS
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
                <img src="data:image/png;base64,{img_str}" width="200" style="margin: 0 auto; display: block;">
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.markdown('<h1 class="main-header">Text Analyzer</h1>', unsafe_allow_html=True)

    st.markdown(
        '<p class="sub-header">POC: Multilingual Emotion & Linguistic Analysis System</p>',
        unsafe_allow_html=True
    )

    # Main tabs
    tabs = st.tabs([
        "🎯 Analysis",
        "📚 Samples",
        "⚙️ Configuration",
        "📖 Documentation",
    ])

    sentiment_colors = {
        "positive": "positive",
        "negative": "negative",
        "ambiguous": "ambiguous",
        "neutral": "neutral",
    }

    # Tab 1: Analysis
    with tabs[0]:
        st.header("Text Analysis")

        analysis_type = st.radio(
            "Analysis Type",
            ["Basic Emotion", "Enhanced (Emotion + Linguistics)"],
            horizontal=True
        )

        # Language and Model Selection
        col_lang, col_model, col_top = st.columns([2, 2, 1])

        with col_lang:
            language = st.selectbox(
                "Language",
                ["English", "Dutch", "Persian"],
                help="Select the language of your text"
            )

        with col_model:
            # Get available models for selected language
            lang_code = language.lower()
            available_models = get_available_models(lang_code)
            model_options = list(available_models.keys())

            if model_options:
                model_key = st.selectbox(
                    "Model",
                    model_options,
                    help=f"Available models for {language}"
                )
            else:
                model_key = "default"
                st.info(f"Using default model for {language}")

        with col_top:
            top_k = st.slider("Top K", 1, 10, 3, help="Number of top emotions to show")

        # Text input
        text_input = st.text_area(
            "Enter text to analyze",
            height=150,
            placeholder="Type or paste your text here...",
        )

        if st.button("🔍 Analyze", type="primary", width='stretch', key="analyze_button"):
            if not text_input:
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner(f"Analyzing with {language} model ({model_key})..."):
                    if analysis_type == "Basic Emotion":
                        analyzer = load_emotion_analyzer(lang_code, model_key)
                        result = analyzer.analyze(text_input, top_k=top_k)

                        # Display model info
                        st.caption(f"🤖 Model: {analyzer.get_model_info()['model_name']}")

                        # Results in a container
                        with st.container(border=True):
                            st.markdown("### 📊 Analysis Results")
                            display_emotion_result(result, sentiment_colors)

                        # Raw scores
                        with st.expander("📊 View All Emotion Scores", expanded=False):
                            scores_df = pd.DataFrame({
                                "Emotion": list(result.emotions.keys()),
                                "Score": list(result.emotions.values()),
                                "Percentage": [f"{v:.2%}" for v in result.emotions.values()]
                            })
                            scores_df = scores_df.sort_values("Score", ascending=False)
                            st.dataframe(
                                scores_df,
                                hide_index=True,
                                width='stretch',
                                column_config={
                                    "Emotion": st.column_config.TextColumn("Emotion", width="medium"),
                                    "Score": st.column_config.ProgressColumn(
                                        "Score",
                                        format="%.4f",
                                        min_value=0,
                                        max_value=1,
                                    ),
                                    "Percentage": st.column_config.TextColumn("Percentage", width="small"),
                                }
                            )

                    else:  # Enhanced
                        analyzer = load_enhanced_analyzer(lang_code, model_key)
                        result = analyzer.analyze(text_input, top_k=top_k)

                        # Display model info
                        st.caption(f"🤖 Model: {analyzer.emotion_analyzer.get_model_info()['model_name']}")

                        # Emotion Analysis in container
                        with st.container(border=True):
                            st.markdown("### 🎭 Emotion Analysis")
                            display_emotion_result(result.emotion_result, sentiment_colors)

                        # Linguistic Analysis in container
                        with st.container(border=True):
                            display_linguistic_features(result.linguistic_features)

                        # Export options
                        st.divider()
                        st.markdown("### 💾 Export Results")

                        from bahar.analyzers.enhanced_analyzer import export_to_academic_format
                        academic_data = export_to_academic_format(result)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                "📥 Download JSON",
                                data=json.dumps(academic_data, indent=2, ensure_ascii=False),
                                file_name="analysis_result.json",
                                mime="application/json",
                                width='stretch',
                            )
                        with col2:
                            # Convert to CSV format
                            csv_lines = ["field,value"]
                            for key, value in academic_data.items():
                                csv_lines.append(f"{key},{value}")
                            st.download_button(
                                "📥 Download CSV",
                                data="\n".join(csv_lines),
                                file_name="analysis_result.csv",
                                mime="text/csv",
                                width='stretch',
                            )
                        with col3:
                            # Show preview
                            with st.popover("👁️ Preview Data"):
                                st.json(academic_data, expanded=False)

    # Tab 2: Samples
    with tabs[1]:
        st.header("📚 Sample Texts")
        st.markdown("Test the analyzer with pre-loaded sample texts in multiple languages.")

        samples = load_samples()

        col1, col2 = st.columns([1, 3])
        with col1:
            sample_lang = st.selectbox("Select Language", list(samples.keys()), key="sample_lang_select")
        with col2:
            st.info(f"📝 {len(samples.get(sample_lang, []))} samples available in {sample_lang}")

        if sample_lang in samples and samples[sample_lang]:
            for idx, sample in enumerate(samples[sample_lang], 1):
                with st.expander(f"📄 Sample {idx}: {sample.get('category', 'general').upper()}", expanded=False):
                    st.markdown("**Text:**")
                    st.info(sample['text'])

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("🔍 Analyze", key=f"sample_{sample_lang}_{idx}"):
                            with st.spinner("Analyzing..."):
                                # Use default model for the language
                                lang_code = sample_lang.lower()
                                analyzer = load_emotion_analyzer(lang_code, "sentiment" if lang_code != "english" else "goemotions")
                                result = analyzer.analyze(sample['text'], top_k=3)

                                with st.container(border=True):
                                    display_emotion_result(result, sentiment_colors)
        else:
            st.warning(f"No samples available for {sample_lang}")

    # Tab 3: Configuration
    with tabs[2]:
        st.header("Configuration")
        st.markdown("Manage taxonomy, emotion groups, and sample texts.")

        config_tabs = st.tabs(["Taxonomy", "Emotion Groups", "Samples"])

        # Taxonomy Configuration
        with config_tabs[0]:
            st.subheader("Emotion Taxonomy")
            st.markdown("List of all emotions in the system.")

            taxonomy = load_taxonomy()

            # Display current taxonomy
            st.markdown(f"**Total Emotions:** {len(taxonomy)}")

            col1, col2 = st.columns([3, 1])
            with col1:
                taxonomy_text = st.text_area(
                    "Emotions (one per line)",
                    value="\n".join(taxonomy),
                    height=300,
                )

            with col2:
                st.markdown("**Actions**")
                if st.button("💾 Save Taxonomy", width='stretch', key="save_taxonomy"):
                    new_taxonomy = [line.strip() for line in taxonomy_text.split("\n") if line.strip()]
                    save_config_file(TAXONOMY_FILE, new_taxonomy)
                    st.success(f"Saved {len(new_taxonomy)} emotions!")
                    st.rerun()

                if st.button("🔄 Reset to Default", width='stretch', key="reset_taxonomy"):
                    save_config_file(TAXONOMY_FILE, list(GOEMOTIONS_EMOTIONS))
                    st.success("Reset to default taxonomy!")
                    st.rerun()

        # Emotion Groups Configuration
        with config_tabs[1]:
            st.subheader("Emotion Groups")
            st.markdown("Organize emotions into sentiment categories.")

            emotion_groups = load_emotion_groups()

            # Display and edit groups
            for group_name, emotions in emotion_groups.items():
                with st.expander(f"{group_name.upper()} ({len(emotions)} emotions)"):
                    emotions_text = st.text_area(
                        f"{group_name} emotions (one per line)",
                        value="\n".join(emotions),
                        height=150,
                        key=f"group_{group_name}",
                    )

                    if st.button(f"Save {group_name}", key=f"save_{group_name}"):
                        emotion_groups[group_name] = [
                            line.strip() for line in emotions_text.split("\n") if line.strip()
                        ]
                        save_config_file(EMOTION_GROUPS_FILE, emotion_groups)
                        st.success(f"Saved {group_name} group!")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save All Groups", type="primary", width='stretch', key="save_all_groups"):
                    save_config_file(EMOTION_GROUPS_FILE, emotion_groups)
                    st.success("All groups saved!")

            with col2:
                if st.button("🔄 Reset to Default", width='stretch', key="reset_groups"):
                    save_config_file(EMOTION_GROUPS_FILE, EMOTION_GROUPS)
                    st.success("Reset to default groups!")
                    st.rerun()

        # Samples Configuration
        with config_tabs[2]:
            st.subheader("Sample Texts")
            st.markdown("Manage sample texts for testing.")

            samples = load_samples()

            # Add new sample
            st.markdown("#### Add New Sample")
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                new_lang = st.text_input("Language", placeholder="e.g., english")
            with col2:
                new_category = st.text_input("Category", placeholder="e.g., positive")
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)

            new_text = st.text_area("Sample Text", placeholder="Enter sample text...")

            if st.button("➕ Add Sample", key="add_sample"):
                if new_lang and new_text:
                    if new_lang not in samples:
                        samples[new_lang] = []
                    samples[new_lang].append({
                        "text": new_text,
                        "category": new_category or "general"
                    })
                    save_config_file(SAMPLES_FILE, samples)
                    st.success(f"Added sample to {new_lang}!")
                    st.rerun()
                else:
                    st.warning("Please provide language and text.")

            st.markdown("---")

            # View/Edit existing samples
            st.markdown("#### Existing Samples")

            edit_lang = st.selectbox("Select Language to Edit", list(samples.keys()))

            if edit_lang in samples:
                samples_json = json.dumps(samples[edit_lang], indent=2, ensure_ascii=False)
                edited_samples = st.text_area(
                    f"{edit_lang} samples (JSON format)",
                    value=samples_json,
                    height=300,
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Changes", width='stretch', key="save_samples"):
                        try:
                            samples[edit_lang] = json.loads(edited_samples)
                            save_config_file(SAMPLES_FILE, samples)
                            st.success(f"Saved {edit_lang} samples!")
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format!")

                with col2:
                    if st.button("🗑️ Delete All", width='stretch', key="delete_samples"):
                        samples[edit_lang] = []
                        save_config_file(SAMPLES_FILE, samples)
                        st.success(f"Deleted all {edit_lang} samples!")
                        st.rerun()

    # Tab 4: Documentation
    with tabs[3]:
        st.header("Documentation")

        doc_tabs = st.tabs(["Overview", "Emotions", "Linguistic Dimensions", "API"])

        with doc_tabs[0]:
            st.markdown("""
            ### 🌸 About Bahar

            **Bahar** is a multilingual emotion and linguistic analysis system that combines:

            - **GoEmotions Dataset**: 28 fine-grained emotion categories
            - **Linguistic Analysis**: Formality, tone, intensity, and communication style
            - **Multilingual Support**: English, Dutch, Persian, and more

            ### 📖 How to Use

            1. **Analysis Tab**: Enter text and choose analysis type
            2. **Samples Tab**: Test with pre-loaded examples
            3. **Configuration Tab**: Customize taxonomy and samples
            4. **Documentation Tab**: Learn about the system

            ### ✨ Features

            - 🎭 28 fine-grained emotions
            - 🌍 Multilingual support
            - 📊 Linguistic analysis
            - 🎨 Professional UI
            - ⚙️ Configurable taxonomy
            - 💾 Export results (JSON/CSV)
            - 📝 Real-time analysis
            - 🔧 Academic research ready

            ### 📌 Version

            **v0.2.0** - Production Ready
            """)

        with doc_tabs[1]:
            st.markdown("### Emotion Categories")

            emotion_groups = load_emotion_groups()

            for group, emotions in emotion_groups.items():
                st.markdown(f"#### {group.upper()}")
                st.markdown(", ".join(emotions))

        with doc_tabs[2]:
            st.markdown("""
            ### Linguistic Dimensions

            #### Formality
            - **Formal**: Professional, academic language
            - **Colloquial**: Casual, everyday language
            - **Neutral**: Standard language

            #### Tone
            - **Friendly**: Warm, approachable
            - **Rough**: Harsh, aggressive
            - **Serious**: Grave, important
            - **Kind**: Gentle, compassionate
            - **Neutral**: Balanced tone

            #### Intensity
            - **High**: Strong emotional expression
            - **Medium**: Moderate expression
            - **Low**: Subtle expression

            #### Communication Style
            - **Direct**: Clear, straightforward
            - **Indirect**: Subtle, implicit
            - **Assertive**: Confident, firm
            - **Passive**: Hesitant, apologetic
            """)

        with doc_tabs[3]:
            st.markdown("""
            ### API Usage

            #### Basic Emotion Analysis
            ```python
            from bahar import EmotionAnalyzer

            analyzer = EmotionAnalyzer(dataset="goemotions")
            analyzer.load_model()
            result = analyzer.analyze("I'm happy!", top_k=3)
            ```

            #### Enhanced Analysis
            ```python
            from bahar import EnhancedAnalyzer

            analyzer = EnhancedAnalyzer(emotion_dataset="goemotions")
            analyzer.load_model()
            result = analyzer.analyze("Your text", top_k=3)
            ```

            #### Export Results
            ```python
            from bahar.analyzers.enhanced_analyzer import export_to_academic_format

            academic_data = export_to_academic_format(result)
            # Returns structured dict for CSV/JSON export
            ```
            """)


if __name__ == "__main__":
    main()

