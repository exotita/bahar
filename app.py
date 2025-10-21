#!/usr/bin/env python3
"""
Bahar - Streamlit Web Application

Professional web interface for multilingual emotion and linguistic analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from bahar.analyzers.emotion_analyzer import EmotionAnalyzer
from bahar.analyzers.enhanced_analyzer import EnhancedAnalyzer
from bahar.datasets.goemotions.taxonomy import EMOTION_GROUPS, GOEMOTIONS_EMOTIONS
from bahar.utils.language_models import get_available_models

if TYPE_CHECKING:
    from bahar.analyzers.advanced_analyzer import AdvancedLinguisticAnalyzer


# Page configuration
st.set_page_config(
    page_title="baarsh - Emotion & Linguistic Analysis",
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
def load_enhanced_analyzer(language: str, model_key: str, enable_nlp: bool = False) -> EnhancedAnalyzer:
    """Load and cache enhanced analyzer for specific language and model."""
    analyzer = EnhancedAnalyzer(language=language, model_key=model_key, enable_nlp=enable_nlp)
    analyzer.load_model()
    return analyzer


@st.cache_resource
def load_advanced_analyzer(
    language: str,
    enable_semantics: bool = True,
    enable_morphology: bool = True,
    enable_embeddings: bool = True,
    enable_discourse: bool = True,
) -> AdvancedLinguisticAnalyzer:
    """
    Load and cache advanced linguistic analyzer.

    Args:
        language: Language code ("english", "dutch")
        enable_semantics: Enable semantic analysis
        enable_morphology: Enable morphology analysis
        enable_embeddings: Enable embedding analysis
        enable_discourse: Enable discourse analysis

    Returns:
        Cached AdvancedLinguisticAnalyzer instance
    """
    from bahar import AdvancedLinguisticAnalyzer

    analyzer = AdvancedLinguisticAnalyzer(
        language=language,
        enable_semantics=enable_semantics,
        enable_morphology=enable_morphology,
        enable_embeddings=enable_embeddings,
        enable_discourse=enable_discourse,
    )
    analyzer.load_models()
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


def display_nlp_features(features) -> None:
    """Display NLP analysis features."""
    st.subheader("NLP Analysis (spaCy)")

    # Text statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tokens", features.num_tokens)
    with col2:
        st.metric("Sentences", features.num_sentences)
    with col3:
        st.metric("Avg Token Length", f"{features.avg_token_length:.1f}")
    with col4:
        st.metric("Lexical Diversity", f"{features.lexical_diversity:.2%}")

    st.divider()

    # Syntactic features
    st.markdown("##### 📊 Syntactic Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Part-of-Speech Distribution:**")
        if features.pos_distribution:
            pos_df = pd.DataFrame({
                "POS": list(features.pos_distribution.keys())[:8],
                "Count": list(features.pos_distribution.values())[:8]
            })
            st.dataframe(pos_df, hide_index=True, width='stretch')

    with col2:
        st.markdown("**Word Categories:**")
        cat_col1, cat_col2 = st.columns(2)
        with cat_col1:
            st.metric("Nouns", features.num_nouns)
            st.metric("Adjectives", features.num_adjectives)
        with cat_col2:
            st.metric("Verbs", features.num_verbs)
            st.metric("Adverbs", features.num_adverbs)
        st.metric("Noun/Verb Ratio", f"{features.noun_verb_ratio:.2f}")

    # Named entities
    if features.entities:
        st.divider()
        st.markdown("##### 🏷️ Named Entities")

        # Entity types summary
        if features.entity_types:
            entity_type_df = pd.DataFrame({
                "Type": list(features.entity_types.keys()),
                "Count": list(features.entity_types.values())
            })
            st.dataframe(entity_type_df, hide_index=True, width='stretch')

        # Entity list
        with st.expander(f"📋 View All Entities ({len(features.entities)})", expanded=False):
            entity_df = pd.DataFrame(features.entities, columns=["Text", "Label", "Description"])
            st.dataframe(entity_df, hide_index=True, width='stretch')

    # Additional features
    st.divider()
    st.markdown("##### ✨ Additional Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Negation", "Yes" if features.has_negation else "No")
    with col2:
        st.metric("Questions", features.question_count)
    with col3:
        st.metric("Exclamations", features.exclamation_count)


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
        '<p class="sub-header">Multilingual Emotion & Linguistic Analysis - Demo Application (POC), Beta Version</p>',
        unsafe_allow_html=True
    )

    # Main tabs
    tabs = st.tabs([
        "🎯 Analysis",
        "🔬 Advanced Analysis",
        "📚 Samples",
        "🤖 Model Management",
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
            ["Basic Emotion", "Enhanced (Emotion + Linguistics)", "Enhanced + NLP (spaCy)", "Advanced Linguistic Analysis"],
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

                # Warning for problematic models
                if model_key == "albert-sentiment":
                    st.warning("⚠️ This model may have tokenizer compatibility issues. Consider using 'sentiment' or 'parsbert-snappfood' instead.")
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

                    elif analysis_type == "Advanced Linguistic Analysis":
                        # Advanced linguistic analysis
                        # Use default parameters for quick analysis
                        analyzer = load_advanced_analyzer(
                            language=lang_code,
                            enable_semantics=True,
                            enable_morphology=True,
                            enable_embeddings=True,
                            enable_discourse=True,
                        )

                        result = analyzer.analyze(
                            text_input,
                            top_k_neighbors=5,
                            n_clusters=3
                        )

                        st.caption("🔬 Advanced Linguistic Analysis (all analyzers enabled)")

                        # Summary metrics
                        summary = result.get_summary()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Text Length", f"{summary['text_length']} chars")
                        with col2:
                            st.metric("Word Count", summary['word_count'])
                        with col3:
                            st.metric("Analyzers", "4 (All)")

                        st.divider()

                        # Semantic Analysis
                        if result.semantic_features:
                            with st.container(border=True):
                                st.markdown("### 📚 Lexical & Compositional Semantics")
                                sem = summary['semantic']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Lexical Diversity", f"{sem['lexical_diversity']:.3f}")
                                with col2:
                                    st.metric("Semantic Density", f"{sem['semantic_density']:.3f}")
                                with col3:
                                    st.metric("Polysemy Rate", f"{sem['polysemy_rate']:.3f}")
                                with col4:
                                    st.metric("Cohesion Score", f"{sem['cohesion_score']:.3f}")

                        # Morphology Analysis
                        if result.morphology_features:
                            with st.container(border=True):
                                st.markdown("### 🔤 Morphology & Phonology")
                                morph = summary['morphology']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Morphemes/Word", f"{morph['morphemes_per_word']:.3f}")
                                with col2:
                                    st.metric("Complexity", f"{morph['morphological_complexity']:.3f}")
                                with col3:
                                    st.metric("Syllables/Word", f"{morph['syllables_per_word']:.3f}")
                                with col4:
                                    st.metric("C/V Ratio", f"{morph['consonant_vowel_ratio']:.3f}")

                        # Embedding Analysis
                        if result.embedding_features:
                            with st.container(border=True):
                                st.markdown("### 🧠 Distributional Semantics & Embeddings")
                                emb = summary['embeddings']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Vector Dims", emb['vector_dimensionality'])
                                with col2:
                                    st.metric("Semantic Density", f"{emb['semantic_density']:.3f}")
                                with col3:
                                    st.metric("Cluster Quality", f"{emb['cluster_quality']:.3f}")
                                with col4:
                                    st.metric("Effective Dims", emb['effective_dimensions'])

                        # Discourse Analysis
                        if result.discourse_features:
                            with st.container(border=True):
                                st.markdown("### 💬 Pragmatics & Discourse")
                                disc = summary['discourse']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Entity Density", f"{disc['entity_density']:.3f}")
                                with col2:
                                    st.metric("Avg Chain Length", f"{disc['avg_chain_length']:.3f}")
                                with col3:
                                    st.metric("Topic Continuity", f"{disc['topic_continuity']:.3f}")
                                with col4:
                                    st.metric("Coherence Score", f"{disc['coherence_score']:.3f}")

                        # Export options
                        st.divider()
                        st.markdown("### 💾 Export Results")

                        academic_data = result.export_academic_format()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                "📥 Download Academic Format (JSON)",
                                data=json.dumps(academic_data, indent=2, ensure_ascii=False),
                                file_name="advanced_analysis_academic.json",
                                mime="application/json",
                                width='stretch',
                            )
                        with col2:
                            complete_data = result.to_dict()
                            st.download_button(
                                "📥 Download Complete Analysis (JSON)",
                                data=json.dumps(complete_data, indent=2, ensure_ascii=False),
                                file_name="advanced_analysis_complete.json",
                                mime="application/json",
                                width='stretch',
                            )
                        with col3:
                            with st.popover("👁️ Preview Data"):
                                st.json(academic_data, expanded=False)

                    else:  # Enhanced or Enhanced + NLP
                        enable_nlp = "NLP" in analysis_type

                        # Show download progress if NLP is enabled
                        if enable_nlp:
                            with st.status("Loading NLP models...", expanded=True) as status:
                                st.write("Loading emotion model...")
                                analyzer = load_enhanced_analyzer(lang_code, model_key, enable_nlp=enable_nlp)

                                st.write("Analyzing text...")
                                result = analyzer.analyze(text_input, top_k=top_k)

                                status.update(label="Analysis complete!", state="complete", expanded=False)
                        else:
                            analyzer = load_enhanced_analyzer(lang_code, model_key, enable_nlp=enable_nlp)
                            result = analyzer.analyze(text_input, top_k=top_k)

                        # Display model info
                        st.caption(f"🤖 Model: {analyzer.emotion_analyzer.get_model_info()['model_name']}")
                        if enable_nlp:
                            if result.nlp_features is not None:
                                st.caption("✓ NLP Analysis enabled (spaCy)")
                            else:
                                st.warning("⚠️ NLP analysis not available. Install spaCy: pip install spacy")

                        # Emotion Analysis in container
                        with st.container(border=True):
                            st.markdown("### 🎭 Emotion Analysis")
                            display_emotion_result(result.emotion_result, sentiment_colors)

                        # Linguistic Analysis in container
                        with st.container(border=True):
                            display_linguistic_features(result.linguistic_features)

                        # NLP Analysis if enabled and available
                        if enable_nlp and result.nlp_features is not None:
                            with st.container(border=True):
                                display_nlp_features(result.nlp_features)

                        # Export options
                        st.divider()
                        st.markdown("### 💾 Export Results")

                        from bahar.analyzers.enhanced_analyzer import export_to_academic_format
                        academic_data = export_to_academic_format(result)

                        # Add NLP data to export if available
                        if result.nlp_features is not None:
                            nlp_summary = result.nlp_features.get_summary()
                            academic_data["nlp_tokens"] = nlp_summary["text_stats"]["tokens"]
                            academic_data["nlp_sentences"] = nlp_summary["text_stats"]["sentences"]
                            academic_data["nlp_lexical_diversity"] = nlp_summary["readability"]["lexical_diversity"]
                            academic_data["nlp_entities_count"] = nlp_summary["entities"]["count"]

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

    # Tab 2: Advanced Analysis
    with tabs[1]:
        st.header("🔬 Advanced Linguistic Analysis")
        st.markdown("Comprehensive linguistic analysis including semantics, morphology, embeddings, and discourse.")

        # Cache info
        st.info("💡 **Performance Tip**: Analyzers are cached for faster repeated analyses with the same configuration.", icon="⚡")

        # Analysis configuration
        st.subheader("Analysis Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Enable Analyzers")
            enable_semantics = st.checkbox("📚 Lexical & Compositional Semantics", value=True,
                                          help="Word sense disambiguation, semantic similarity, lexical chains")
            enable_morphology = st.checkbox("🔤 Morphology & Phonology", value=True,
                                           help="Morpheme segmentation, syllabification, phonological features")
            enable_embeddings = st.checkbox("🧠 Distributional Semantics & Embeddings", value=True,
                                           help="Word embeddings, semantic clustering, dimensionality analysis")
            enable_discourse = st.checkbox("💬 Pragmatics & Discourse", value=True,
                                          help="Entity tracking, coreference, discourse relations, coherence")

        with col2:
            st.markdown("##### Language & Parameters")
            adv_language = st.selectbox(
                "Language",
                ["English", "Dutch"],
                key="adv_lang_select",
                help="Select analysis language"
            )

            top_k_neighbors = st.slider(
                "Semantic Neighbors (k)",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of semantic neighbors to find"
            )

            n_clusters = st.slider(
                "Semantic Clusters",
                min_value=2,
                max_value=5,
                value=3,
                help="Number of clusters for semantic space"
            )

        st.divider()

        # Text input with sample texts
        st.subheader("Text Input")

        # Sample texts for testing
        col_input, col_samples = st.columns([3, 1])

        with col_samples:
            st.markdown("**Quick Test**")
            sample_texts = {
                "Academic (EN)": "Natural language processing enables computers to understand, interpret, and generate human language. Modern NLP systems use deep learning architectures and large language models trained on vast corpora.",
                "Narrative (EN)": "The old lighthouse stood on the rocky cliff, its beam cutting through the fog. For decades, it had guided ships safely to harbor, a silent guardian watching over the restless sea.",
                "Technical (EN)": "The algorithm implements a recursive backtracking approach with memoization to optimize performance. Time complexity is O(n²) in the worst case, with space complexity of O(n).",
                "Conversational (EN)": "Hey! I just wanted to let you know that the meeting went really well. Everyone loved your presentation, and we got approval for the project. Can't wait to get started!",
                "News (EN)": "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. The creature, found at depths exceeding 8,000 meters, exhibits unique bioluminescent properties.",
                "Academic (NL)": "Kunstmatige intelligentie heeft een revolutie teweeggebracht in de manier waarop we technologie gebruiken. Machine learning-algoritmen kunnen patronen herkennen en voorspellingen doen op basis van grote datasets.",
                "Narrative (NL)": "De oude molen stond er al eeuwen, zijn wieken draaiend in de wind. Generaties hadden er hun graan gemalen, en de molen was een symbool geworden van het dorp.",
            }

            selected_sample = st.selectbox(
                "Load Sample",
                ["Custom"] + list(sample_texts.keys()),
                key="adv_sample_select",
                help="Select a sample text to test"
            )

            # Update text when sample changes
            if selected_sample != "Custom":
                if st.button("📥 Load", key="load_adv_sample", width='stretch', type="secondary"):
                    st.session_state.adv_text_input = sample_texts[selected_sample]
                    st.rerun()

        with col_input:
            # Initialize session state if not exists
            if "adv_text_input" not in st.session_state:
                st.session_state.adv_text_input = ""

            adv_text_input = st.text_area(
                "Enter text for advanced analysis",
                value=st.session_state.adv_text_input,
                height=150,
                placeholder="Enter your text here for comprehensive linguistic analysis...",
                key="adv_text_input"
            )

        # Analyze button
        if st.button("🔬 Perform Advanced Analysis", type="primary", width='stretch', key="adv_analyze_button"):
            if not adv_text_input:
                st.warning("Please enter some text to analyze.")
            elif not any([enable_semantics, enable_morphology, enable_embeddings, enable_discourse]):
                st.warning("Please enable at least one analyzer.")
            else:
                lang_code = adv_language.lower()

                with st.spinner("Performing advanced linguistic analysis..."):
                    try:
                        # Load cached analyzer
                        analyzer = load_advanced_analyzer(
                            language=lang_code,
                            enable_semantics=enable_semantics,
                            enable_morphology=enable_morphology,
                            enable_embeddings=enable_embeddings,
                            enable_discourse=enable_discourse,
                        )

                        # Perform analysis
                        result = analyzer.analyze(
                            adv_text_input,
                            top_k_neighbors=top_k_neighbors,
                            n_clusters=n_clusters
                        )

                        st.success("✅ Analysis complete!")

                        # Display results
                        st.divider()
                        st.subheader("📊 Analysis Results")

                        # Summary metrics
                        summary = result.get_summary()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Text Length", f"{summary['text_length']} chars")
                        with col2:
                            st.metric("Word Count", summary['word_count'])
                        with col3:
                            enabled_count = sum([enable_semantics, enable_morphology, enable_embeddings, enable_discourse])
                            st.metric("Analyzers Used", enabled_count)

                        st.divider()

                        # Semantic Analysis Results
                        if enable_semantics and result.semantic_features:
                            with st.container(border=True):
                                st.markdown("### 📚 Lexical & Compositional Semantics")

                                sem = summary['semantic']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Lexical Diversity", f"{sem['lexical_diversity']:.3f}")
                                with col2:
                                    st.metric("Semantic Density", f"{sem['semantic_density']:.3f}")
                                with col3:
                                    st.metric("Polysemy Rate", f"{sem['polysemy_rate']:.3f}")
                                with col4:
                                    st.metric("Cohesion Score", f"{sem['cohesion_score']:.3f}")

                                # Word senses
                                if result.semantic_features.word_senses:
                                    with st.expander(f"📖 Word Senses ({len(result.semantic_features.word_senses)} words)", expanded=False):
                                        for ws in result.semantic_features.word_senses[:10]:  # Show first 10
                                            st.markdown(f"**{ws.word}** ({ws.pos})")
                                            st.caption(f"Definition: {ws.definition}")
                                            if ws.examples:
                                                st.caption(f"Example: {ws.examples[0]}")
                                            st.caption(f"Confidence: {ws.confidence:.2f}")
                                            st.divider()

                                # Lexical chains
                                if result.semantic_features.lexical_chains:
                                    with st.expander(f"🔗 Lexical Chains ({len(result.semantic_features.lexical_chains)})", expanded=False):
                                        for chain in result.semantic_features.lexical_chains:
                                            st.markdown(f"**Type:** {chain.chain_type}")
                                            st.markdown(f"**Words:** {', '.join(chain.words[:10])}")
                                            st.caption(f"Strength: {chain.strength:.3f} | Length: {len(chain.words)}")
                                            st.divider()

                        # Morphology Analysis Results
                        if enable_morphology and result.morphology_features:
                            with st.container(border=True):
                                st.markdown("### 🔤 Morphology & Phonology")

                                morph = summary['morphology']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Morphemes/Word", f"{morph['morphemes_per_word']:.3f}")
                                with col2:
                                    st.metric("Morphological Complexity", f"{morph['morphological_complexity']:.3f}")
                                with col3:
                                    st.metric("Syllables/Word", f"{morph['syllables_per_word']:.3f}")
                                with col4:
                                    st.metric("C/V Ratio", f"{morph['consonant_vowel_ratio']:.3f}")

                                # Morpheme details
                                if result.morphology_features.morpheme_analyses:
                                    with st.expander(f"🔍 Morpheme Analysis ({len(result.morphology_features.morpheme_analyses)} words)", expanded=False):
                                        for ma in result.morphology_features.morpheme_analyses[:10]:
                                            st.markdown(f"**{ma.word}** → {ma.lemma}")
                                            st.caption(f"Morphemes: {' + '.join(ma.morphemes)}")
                                            if ma.affixes:
                                                st.caption(f"Affixes: {', '.join(ma.affixes)}")
                                            st.divider()

                        # Embedding Analysis Results
                        if enable_embeddings and result.embedding_features:
                            with st.container(border=True):
                                st.markdown("### 🧠 Distributional Semantics & Embeddings")

                                emb = summary['embeddings']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Vector Dimensions", emb['vector_dimensionality'])
                                with col2:
                                    st.metric("Semantic Density", f"{emb['semantic_density']:.3f}")
                                with col3:
                                    st.metric("Cluster Quality", f"{emb['cluster_quality']:.3f}")
                                with col4:
                                    st.metric("Effective Dimensions", emb['effective_dimensions'])

                                # Semantic neighbors
                                if result.embedding_features.semantic_neighbors:
                                    with st.expander(f"🎯 Semantic Neighbors ({len(result.embedding_features.semantic_neighbors)})", expanded=False):
                                        # Group by target word
                                        neighbors_by_word = {}
                                        for neighbor in result.embedding_features.semantic_neighbors:
                                            if neighbor.target_word not in neighbors_by_word:
                                                neighbors_by_word[neighbor.target_word] = []
                                            neighbors_by_word[neighbor.target_word].append(neighbor)

                                        for target_word, neighbors in list(neighbors_by_word.items())[:5]:
                                            st.markdown(f"**{target_word}:**")
                                            for n in neighbors:
                                                st.caption(f"  → {n.neighbor_word} (similarity: {n.similarity:.3f})")
                                            st.divider()

                                # Clusters
                                if result.embedding_features.clusters:
                                    with st.expander(f"📊 Semantic Clusters ({len(result.embedding_features.clusters)})", expanded=False):
                                        for cluster in result.embedding_features.clusters:
                                            st.markdown(f"**Cluster {cluster.cluster_id + 1}** (cohesion: {cluster.cohesion:.3f})")
                                            st.caption(f"Words: {', '.join(cluster.words[:15])}")
                                            st.divider()

                        # Discourse Analysis Results
                        if enable_discourse and result.discourse_features:
                            with st.container(border=True):
                                st.markdown("### 💬 Pragmatics & Discourse")

                                disc = summary['discourse']

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Entity Density", f"{disc['entity_density']:.3f}")
                                with col2:
                                    st.metric("Avg Chain Length", f"{disc['avg_chain_length']:.3f}")
                                with col3:
                                    st.metric("Topic Continuity", f"{disc['topic_continuity']:.3f}")
                                with col4:
                                    st.metric("Coherence Score", f"{disc['coherence_score']:.3f}")

                                # Entities
                                if result.discourse_features.entity_mentions:
                                    with st.expander(f"🏷️ Entity Mentions ({len(result.discourse_features.entity_mentions)})", expanded=False):
                                        entity_df = pd.DataFrame([
                                            {"Entity": em.text, "Type": em.label, "Sentence": em.sentence_id}
                                            for em in result.discourse_features.entity_mentions[:20]
                                        ])
                                        st.dataframe(entity_df, hide_index=True, width='stretch')

                                # Coreference chains
                                if result.discourse_features.coreference_chains:
                                    with st.expander(f"🔗 Coreference Chains ({len(result.discourse_features.coreference_chains)})", expanded=False):
                                        for chain in result.discourse_features.coreference_chains:
                                            st.markdown(f"**Chain {chain.chain_id + 1}** ({chain.chain_type})")
                                            mentions = [m.text for m in chain.mentions]
                                            st.caption(f"Mentions: {' → '.join(mentions)}")
                                            st.divider()

                                # Information flow
                                if result.discourse_features.information_flow:
                                    with st.expander(f"📈 Information Flow ({len(result.discourse_features.information_flow)} sentences)", expanded=False):
                                        for flow in result.discourse_features.information_flow[:10]:
                                            st.markdown(f"**Sentence {flow.sentence_id + 1}**")
                                            if flow.topic:
                                                st.caption(f"Topic: {flow.topic}")
                                            if flow.new_entities:
                                                st.caption(f"New: {', '.join(flow.new_entities)}")
                                            if flow.given_entities:
                                                st.caption(f"Given: {', '.join(flow.given_entities)}")
                                            st.divider()

                        # Export section
                        st.divider()
                        st.subheader("💾 Export Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Academic format export
                            academic_data = result.export_academic_format()
                            academic_json = json.dumps(academic_data, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📊 Download Academic Format (JSON)",
                                data=academic_json,
                                file_name="advanced_analysis_academic.json",
                                mime="application/json",
                                width='stretch'
                            )

                        with col2:
                            # Complete format export
                            complete_data = result.to_dict()
                            complete_json = json.dumps(complete_data, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📄 Download Complete Analysis (JSON)",
                                data=complete_json,
                                file_name="advanced_analysis_complete.json",
                                mime="application/json",
                                width='stretch'
                            )

                        # Show preview
                        with st.expander("👁️ Preview Academic Format", expanded=False):
                            st.json(academic_data, expanded=False)

                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())

    # Tab 3: Samples
    with tabs[2]:
        st.header("📚 Sample Texts")
        st.markdown("Test the analyzer with pre-loaded sample texts in multiple languages.")

        samples = load_samples()

        # Analysis configuration
        st.subheader("Analysis Configuration")

        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            sample_lang = st.selectbox(
                "Language",
                list(samples.keys()),
                key="sample_lang_select",
                help="Select language for samples"
            )

        with col2:
            # Get available models for selected language
            lang_code = sample_lang.lower()
            available_models = get_available_models(lang_code)
            model_options = list(available_models.keys())

            if model_options:
                sample_model_key = st.selectbox(
                    "Model",
                    model_options,
                    key="sample_model_select",
                    help=f"Available models for {sample_lang}"
                )
            else:
                sample_model_key = "default"
                st.info("Using default model")

        with col3:
            sample_analysis_type = st.selectbox(
                "Analysis Type",
                ["Basic Emotion", "Enhanced (Emotion + Linguistics)", "Enhanced + NLP (spaCy)"],
                key="sample_analysis_type",
                help="Choose analysis type"
            )

        with col4:
            sample_top_k = st.slider(
                "Top K",
                1, 10, 3,
                key="sample_top_k",
                help="Number of top emotions"
            )

        st.divider()

        # Display samples
        if sample_lang in samples and samples[sample_lang]:
            st.markdown(f"**{len(samples[sample_lang])} sample(s) available**")

            for idx, sample in enumerate(samples[sample_lang], 1):
                with st.expander(f"📄 Sample {idx}: {sample.get('category', 'general').upper()}", expanded=False):
                    st.markdown("**Text:**")
                    st.info(sample['text'])

                    if st.button("🔍 Analyze", key=f"sample_{sample_lang}_{idx}", width='stretch'):
                        with st.spinner(f"Analyzing with {sample_analysis_type}..."):
                            try:
                                if sample_analysis_type == "Basic Emotion":
                                    # Basic emotion analysis
                                    analyzer = load_emotion_analyzer(lang_code, sample_model_key)
                                    result = analyzer.analyze(sample['text'], top_k=sample_top_k)

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

                                else:  # Enhanced or Enhanced + NLP
                                    # Enhanced analysis
                                    enable_nlp = "NLP" in sample_analysis_type

                                    # Show download progress if NLP is enabled
                                    if enable_nlp:
                                        with st.status("Loading NLP models...", expanded=True) as status:
                                            st.write("Loading emotion model...")
                                            analyzer = load_enhanced_analyzer(lang_code, sample_model_key, enable_nlp=enable_nlp)

                                            st.write("Analyzing text...")
                                            result = analyzer.analyze(sample['text'], top_k=sample_top_k)

                                            status.update(label="Analysis complete!", state="complete", expanded=False)
                                    else:
                                        analyzer = load_enhanced_analyzer(lang_code, sample_model_key, enable_nlp=enable_nlp)
                                        result = analyzer.analyze(sample['text'], top_k=sample_top_k)

                                    # Display model info
                                    st.caption(f"🤖 Model: {analyzer.emotion_analyzer.get_model_info()['model_name']}")
                                    if enable_nlp:
                                        if result.nlp_features is not None:
                                            st.caption("✓ NLP Analysis enabled (spaCy)")
                                        else:
                                            st.warning("⚠️ NLP analysis not available. Install spaCy: pip install spacy")

                                    # Emotion Analysis in container
                                    with st.container(border=True):
                                        st.markdown("### 🎭 Emotion Analysis")
                                        display_emotion_result(result.emotion_result, sentiment_colors)

                                    # Linguistic Analysis in container
                                    with st.container(border=True):
                                        display_linguistic_features(result.linguistic_features)

                                    # NLP Analysis if enabled and available
                                    if enable_nlp and result.nlp_features is not None:
                                        with st.container(border=True):
                                            display_nlp_features(result.nlp_features)

                                    # Export options
                                    st.divider()
                                    st.markdown("### 💾 Export Results")

                                    from bahar.analyzers.enhanced_analyzer import export_to_academic_format
                                    academic_data = export_to_academic_format(result)

                                    # Add NLP data to export if available
                                    if result.nlp_features is not None:
                                        nlp_summary = result.nlp_features.get_summary()
                                        academic_data["nlp_tokens"] = nlp_summary["text_stats"]["tokens"]
                                        academic_data["nlp_sentences"] = nlp_summary["text_stats"]["sentences"]
                                        academic_data["nlp_lexical_diversity"] = nlp_summary["readability"]["lexical_diversity"]
                                        academic_data["nlp_entities_count"] = nlp_summary["entities"]["count"]

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.download_button(
                                            "📥 Download JSON",
                                            data=json.dumps(academic_data, indent=2, ensure_ascii=False),
                                            file_name=f"sample_{idx}_analysis.json",
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
                                            file_name=f"sample_{idx}_analysis.csv",
                                            mime="text/csv",
                                            width='stretch',
                                        )
                                    with col3:
                                        # Show preview
                                        with st.popover("👁️ Preview Data"):
                                            st.json(academic_data, expanded=False)

                            except Exception as e:
                                st.error(f"Error during analysis: {e}")
                                st.exception(e)
        else:
            st.warning(f"No samples available for {sample_lang}")

    # Tab 4: Model Management
    with tabs[3]:
        st.header("🤖 Model Management")
        st.markdown("Add, manage, and test models from HuggingFace Hub dynamically.")

        # Import universal loader components
        from bahar.models import (
            ModelRegistry,
            UniversalModelLoader,
            ModelInspector,
            UniversalAdapter,
            ModelMetadata,
        )

        # Initialize registry
        registry = ModelRegistry()

        # Sub-tabs for model management
        model_tabs = st.tabs(["📋 Model List", "➕ Add Model", "🧪 Test Model"])

        # Tab 3.1: Model List
        with model_tabs[0]:
            st.subheader("Registered Models")

            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_task = st.selectbox(
                    "Filter by Task",
                    ["All", "text-classification", "token-classification"],
                    key="filter_task"
                )
            with col2:
                filter_lang = st.selectbox(
                    "Filter by Language",
                    ["All", "english", "dutch", "persian"],
                    key="filter_lang"
                )
            with col3:
                filter_taxonomy = st.selectbox(
                    "Filter by Taxonomy",
                    ["All", "goemotions", "sentiment", "custom"],
                    key="filter_taxonomy"
                )

            # Get filtered models
            models = registry.list_models(
                task_type=None if filter_task == "All" else filter_task,
                language=None if filter_lang == "All" else filter_lang,
                taxonomy=None if filter_taxonomy == "All" else filter_taxonomy,
            )

            if models:
                st.markdown(f"**Found {len(models)} model(s)**")

                for model_meta in models:
                    with st.expander(f"📦 {model_meta.name}", expanded=False):
                        # Check if we're in edit mode for this model
                        edit_key = f"edit_mode_{model_meta.model_id}"
                        if edit_key not in st.session_state:
                            st.session_state[edit_key] = False

                        if not st.session_state[edit_key]:
                            # View Mode
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown(f"**Model ID:** `{model_meta.model_id}`")
                                st.markdown(f"**Description:** {model_meta.description}")
                                st.markdown(f"**Task:** {model_meta.task_type}")
                                st.markdown(f"**Taxonomy:** {model_meta.taxonomy}")
                                st.markdown(f"**Labels:** {model_meta.num_labels}")

                                # Language display
                                if isinstance(model_meta.language, list):
                                    langs = ", ".join(model_meta.language)
                                else:
                                    langs = model_meta.language
                                st.markdown(f"**Languages:** {langs}")

                                # Tags
                                if model_meta.tags:
                                    st.markdown(f"**Tags:** {', '.join(model_meta.tags)}")

                                # Show labels
                                with st.expander("📋 View Labels", expanded=False):
                                    label_df = pd.DataFrame({
                                        "Index": list(model_meta.label_map.keys()),
                                        "Label": list(model_meta.label_map.values())
                                    })
                                    st.dataframe(label_df, hide_index=True, width='stretch')

                            with col2:
                                st.metric("Use Count", model_meta.use_count)
                                if model_meta.last_used:
                                    st.caption(f"Last used: {model_meta.last_used.strftime('%Y-%m-%d %H:%M')}")

                                # Actions
                                col_edit, col_remove = st.columns(2)
                                with col_edit:
                                    if st.button("✏️ Edit", key=f"edit_btn_{model_meta.model_id}", width='stretch'):
                                        st.session_state[edit_key] = True
                                        st.rerun()

                                with col_remove:
                                    if st.button("🗑️ Remove", key=f"remove_{model_meta.model_id}", width='stretch'):
                                        try:
                                            registry.remove_model(model_meta.model_id)
                                            st.success(f"Removed {model_meta.name}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error: {e}")

                        else:
                            # Edit Mode
                            st.markdown("### ✏️ Edit Model Details")

                            with st.form(key=f"edit_form_{model_meta.model_id}"):
                                # Basic Info
                                edit_name = st.text_input("Display Name", value=model_meta.name)
                                edit_description = st.text_area("Description", value=model_meta.description, height=100)

                                col1, col2 = st.columns(2)
                                with col1:
                                    edit_taxonomy = st.selectbox(
                                        "Taxonomy",
                                        ["goemotions", "sentiment", "custom", "star_rating", "binary"],
                                        index=["goemotions", "sentiment", "custom", "star_rating", "binary"].index(model_meta.taxonomy) if model_meta.taxonomy in ["goemotions", "sentiment", "custom", "star_rating", "binary"] else 2
                                    )
                                with col2:
                                    # Language input
                                    if isinstance(model_meta.language, list):
                                        lang_str = ", ".join(model_meta.language)
                                    else:
                                        lang_str = model_meta.language
                                    edit_languages = st.text_input("Languages (comma-separated)", value=lang_str)

                                # Tags
                                edit_tags = st.text_input("Tags (comma-separated)", value=", ".join(model_meta.tags) if model_meta.tags else "")

                                # Labels Editor
                                st.markdown("#### 📋 Edit Labels")
                                st.caption("Edit the label mappings below. Format: index -> label name")

                                # Create editable label mapping
                                label_text = "\n".join([f"{idx}: {label}" for idx, label in model_meta.label_map.items()])
                                edit_labels = st.text_area(
                                    "Labels (one per line, format: index: label)",
                                    value=label_text,
                                    height=200,
                                    help="Each line should be in format: 0: label_name"
                                )

                                # Form buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    submit_edit = st.form_submit_button("💾 Save Changes", type="primary", width='stretch')
                                with col2:
                                    cancel_edit = st.form_submit_button("❌ Cancel", width='stretch')

                            if cancel_edit:
                                st.session_state[edit_key] = False
                                st.rerun()

                            if submit_edit:
                                try:
                                    # Parse labels
                                    new_label_map = {}
                                    for line in edit_labels.strip().split("\n"):
                                        if ":" in line:
                                            idx_str, label = line.split(":", 1)
                                            new_label_map[int(idx_str.strip())] = label.strip()

                                    # Parse languages
                                    new_languages = [lang.strip() for lang in edit_languages.split(",") if lang.strip()]

                                    # Parse tags
                                    new_tags = [tag.strip() for tag in edit_tags.split(",") if tag.strip()]

                                    # Update metadata
                                    model_meta.name = edit_name
                                    model_meta.description = edit_description
                                    model_meta.taxonomy = edit_taxonomy
                                    model_meta.language = new_languages
                                    model_meta.tags = new_tags
                                    model_meta.label_map = new_label_map
                                    model_meta.num_labels = len(new_label_map)

                                    # Save to registry
                                    registry.update_model(model_meta)

                                    st.success(f"✓ Updated {edit_name}")
                                    st.session_state[edit_key] = False
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Error updating model: {e}")
                                    st.exception(e)
            else:
                st.info("No models found. Add a model using the 'Add Model' tab.")

        # Tab 3.2: Add Model
        with model_tabs[1]:
            st.subheader("Add New Model from HuggingFace")

            st.markdown("""
            Enter a HuggingFace model ID to add it to the registry. The system will automatically:
            - Load and inspect the model
            - Detect capabilities and labels
            - Identify taxonomy type
            - Create metadata
            """)

            # Model input form
            with st.form("add_model_form"):
                model_id = st.text_input(
                    "HuggingFace Model ID",
                    placeholder="e.g., cardiffnlp/twitter-roberta-base-emotion",
                    help="Enter the full HuggingFace model identifier"
                )

                col1, col2 = st.columns(2)
                with col1:
                    model_name = st.text_input(
                        "Display Name",
                        placeholder="e.g., Twitter RoBERTa Emotion",
                        help="Human-readable name for the model"
                    )
                with col2:
                    model_tags = st.text_input(
                        "Tags (comma-separated)",
                        placeholder="e.g., emotion, twitter, roberta",
                        help="Tags for categorization and search"
                    )

                model_description = st.text_area(
                    "Description",
                    placeholder="Brief description of the model...",
                    help="Describe what the model does"
                )

                submit_button = st.form_submit_button("🔍 Load and Add Model", type="primary")

            if submit_button and model_id:
                with st.spinner(f"Loading model '{model_id}'..."):
                    try:
                        # Initialize loader
                        loader = UniversalModelLoader()

                        # Validate model first
                        is_valid, error_msg = loader.validate_model(model_id)
                        if not is_valid:
                            st.error(f"Model validation failed: {error_msg}")
                        else:
                            # Load model
                            model, tokenizer, config = loader.load_model(model_id)
                            st.success("✓ Model loaded successfully!")

                            # Inspect model
                            capabilities = ModelInspector.inspect_model(model, tokenizer, config)
                            labels = ModelInspector.extract_labels(config)
                            taxonomy = ModelInspector.detect_taxonomy(labels)
                            languages = ModelInspector.get_supported_languages(model_id)

                            # Display detected information
                            st.markdown("### 🔍 Detected Information")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Task Type", capabilities.task_type)
                                st.metric("Num Labels", capabilities.num_labels)
                            with col2:
                                st.metric("Taxonomy", taxonomy)
                                st.metric("Architecture", capabilities.architecture or "Unknown")
                            with col3:
                                st.metric("Max Length", capabilities.max_length)
                                st.metric("Languages", len(languages))

                            # Show sample labels
                            with st.expander("📋 View Labels", expanded=False):
                                label_df = pd.DataFrame({
                                    "Index": list(labels.keys()),
                                    "Label": list(labels.values())
                                })
                                st.dataframe(label_df, hide_index=True, width='stretch')

                            # Create metadata
                            metadata = ModelMetadata(
                                model_id=model_id,
                                name=model_name or model_id.split("/")[-1],
                                description=model_description or f"Model from HuggingFace: {model_id}",
                                task_type=capabilities.task_type,
                                language=languages,
                                num_labels=capabilities.num_labels,
                                label_map=labels,
                                taxonomy=taxonomy,
                                tags=[tag.strip() for tag in model_tags.split(",")] if model_tags else [],
                            )

                            # Add to registry
                            try:
                                registry.add_model(metadata)
                                st.success(f"✓ Model '{metadata.name}' added to registry!")
                                st.balloons()
                            except ValueError as e:
                                st.warning(f"Model already exists: {e}")

                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        st.exception(e)

            elif submit_button:
                st.warning("Please enter a model ID")

        # Tab 3.3: Test Model
        with model_tabs[2]:
            st.subheader("Test a Model")

            models = registry.list_models()

            if not models:
                st.info("No models available. Add a model first.")
            else:
                # Model selection
                model_options = {m.name: m.model_id for m in models}
                selected_name = st.selectbox(
                    "Select Model to Test",
                    list(model_options.keys()),
                    key="test_model_select"
                )

                selected_id = model_options[selected_name]
                selected_metadata = registry.get_model(selected_id)

                if selected_metadata:
                    # Display model info
                    with st.expander("ℹ️ Model Information", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**ID:** `{selected_metadata.model_id}`")
                            st.markdown(f"**Task:** {selected_metadata.task_type}")
                            st.markdown(f"**Taxonomy:** {selected_metadata.taxonomy}")
                        with col2:
                            st.markdown(f"**Labels:** {selected_metadata.num_labels}")
                            st.markdown(f"**Use Count:** {selected_metadata.use_count}")

                    # Test input
                    test_text = st.text_area(
                        "Enter text to analyze",
                        height=100,
                        placeholder="Type or paste text here...",
                        key="test_text"
                    )

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        top_k = st.slider("Top K", 1, 10, 3, key="test_top_k")

                    if st.button("🔍 Analyze", type="primary", key="test_analyze"):
                        if test_text:
                            with st.spinner("Analyzing..."):
                                try:
                                    # Load model
                                    loader = UniversalModelLoader()
                                    model, tokenizer, config = loader.load_from_metadata(selected_metadata)

                                    # Create adapter
                                    adapter = UniversalAdapter(model, tokenizer, selected_metadata)

                                    # Predict
                                    result = adapter.predict(test_text, top_k=top_k)

                                    # Update usage
                                    registry.update_usage(selected_id)

                                    # Display results
                                    st.markdown("### 📊 Results")

                                    # Top predictions
                                    with st.container(border=True):
                                        st.markdown("#### Top Predictions")

                                        for i, (label, score) in enumerate(result.top_predictions, 1):
                                            col1, col2, col3 = st.columns([2, 2, 1])
                                            with col1:
                                                st.markdown(f"**{i}. {label}**")
                                            with col2:
                                                st.progress(score, text=f"{score:.2%}")
                                            with col3:
                                                st.metric("", f"{score:.3f}")

                                    # All predictions
                                    with st.expander("📋 View All Predictions", expanded=False):
                                        pred_df = pd.DataFrame({
                                            "Label": list(result.predictions.keys()),
                                            "Score": list(result.predictions.values()),
                                            "Percentage": [f"{v:.2%}" for v in result.predictions.values()]
                                        })
                                        pred_df = pred_df.sort_values("Score", ascending=False)
                                        st.dataframe(
                                            pred_df,
                                            hide_index=True,
                                            width='stretch',
                                            column_config={
                                                "Score": st.column_config.ProgressColumn(
                                                    "Score",
                                                    format="%.4f",
                                                    min_value=0,
                                                    max_value=1,
                                                ),
                                            }
                                        )

                                    # Export
                                    st.markdown("### 💾 Export")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.download_button(
                                            "📥 Download JSON",
                                            data=result.to_json(),
                                            file_name="prediction_result.json",
                                            mime="application/json",
                                            width='stretch'
                                        )
                                    with col2:
                                        with st.popover("👁️ Preview JSON"):
                                            st.json(result.to_dict(), expanded=False)

                                except Exception as e:
                                    st.error(f"Error during analysis: {e}")
                                    st.exception(e)
                        else:
                            st.warning("Please enter some text to analyze")

    # Tab 5: Configuration
    with tabs[4]:
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

    # Tab 6: Documentation
    with tabs[5]:
        st.header("Documentation")

        doc_tabs = st.tabs(["Overview", "Emotions", "Linguistic Dimensions", "Advanced Analysis", "Model Management", "API"])

        # Documentation directory
        docs_dir = Path("docs/app")

        with doc_tabs[0]:
            # Load overview documentation
            overview_file = docs_dir / "overview.md"
            if overview_file.exists():
                with open(overview_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.error("Overview documentation not found.")

        with doc_tabs[1]:
            # Load emotions documentation
            emotions_file = docs_dir / "emotions.md"
            if emotions_file.exists():
                with open(emotions_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.error("Emotions documentation not found.")

        with doc_tabs[2]:
            # Load linguistic dimensions documentation
            linguistic_file = docs_dir / "linguistic-dimensions.md"
            if linguistic_file.exists():
                with open(linguistic_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.error("Linguistic dimensions documentation not found.")

        with doc_tabs[3]:
            # Load advanced analysis documentation
            advanced_file = docs_dir / "advanced-analysis.md"
            if advanced_file.exists():
                with open(advanced_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.error("Advanced analysis documentation not found.")

        with doc_tabs[4]:
            # Load model management documentation
            model_mgmt_file = docs_dir / "model-management.md"
            if model_mgmt_file.exists():
                with open(model_mgmt_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.error("Model management documentation not found.")

        with doc_tabs[5]:
            # Load API usage documentation
            api_file = docs_dir / "api-usage.md"
            if api_file.exists():
                with open(api_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            else:
                st.error("API usage documentation not found.")


if __name__ == "__main__":
    main()

