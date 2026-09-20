import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd


# ==========================================================
# PROJECT PATHS
# ==========================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ==========================================================
# IMPORT PIPELINE
# ==========================================================

try:
    from src.pipeline import AudioContextPipeline
except ImportError:
    from pipeline import AudioContextPipeline  # type: ignore[import-not-found]


# ==========================================================
# PATHS
# ==========================================================

CONTEXT_AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_audio"
)

CONTEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contexts"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "test.json"
)

EVALUATION_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation_results.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "audio_event_classifier.joblib"
)


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Audio Context Layer",
    page_icon="🎧",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #4b5563;
        background-color: #1f2937;
        color: #ffffff !important;
        font-size: 1.15rem;
        line-height: 1.6;
    }

    .event-card {
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD PIPELINE
# ==========================================================

@st.cache_resource
def load_pipeline():
    return AudioContextPipeline()


pipeline = load_pipeline()


# ==========================================================
# LOAD CONTEXT LIST
# ==========================================================

@st.cache_data
def get_context_ids():

    files = sorted(
        CONTEXT_DIR.glob("context_*.json")
    )

    return [
        file.stem
        for file in files
    ]


# ==========================================================
# LOAD CONTEXT
# ==========================================================

@st.cache_data
def load_context(context_id):

    context_file = (
        CONTEXT_DIR
        / f"{context_id}.json"
    )

    if not context_file.exists():
        return None

    with open(
        context_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================================
# LOAD TEST DATA
# ==========================================================

@st.cache_data
def load_test_data():

    if not TEST_FILE.exists():
        return []

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================================
# LOAD EVALUATION
# ==========================================================

@st.cache_data
def load_evaluation():

    if not EVALUATION_FILE.exists():
        return None

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================================
# HELPER: FORMAT PERCENTAGE
# ==========================================================

def percent(value):

    return f"{value * 100:.2f}%"


# ==========================================================
# HELPER: EVENT DISPLAY
# ==========================================================

def display_event_timeline(context):

    events = context.get(
        "events",
        []
    )

    if not events:

        st.info(
            "No events were detected."
        )

        return

    rows = []

    for index, event in enumerate(events, 1):

        confidence = event.get(
            "confidence"
        )

        rows.append(
            {
                "Event #": index,
                "Sound": event.get(
                    "name",
                    "unknown"
                ),
                "Start (s)": round(
                    event.get("start", 0),
                    2
                ),
                "End (s)": round(
                    event.get("end", 0),
                    2
                ),
                "Duration (s)": round(
                    (
                        event.get("end", 0)
                        - event.get("start", 0)
                    ),
                    2
                ),
                "Confidence": (
                    round(confidence, 3)
                    if confidence is not None
                    else None
                )
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🎧 Audio Context Layer")

st.sidebar.markdown(
    """
    **Audio QA Proof of Concept**

    The system converts audio into a structured
    temporal context layer and answers questions
    grounded in detected sound events.
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Audio QA Demo",
        "Context Explorer",
        "Evaluation"
    ]
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🎧 Audio Context Layer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Structured audio understanding with temporal context,
    question classification, retrieval and answer generation.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# AUDIO QA DEMO
# ==========================================================

if page == "Audio QA Demo":

    st.header("Ask Questions About Audio")

    context_ids = get_context_ids()

    if not context_ids:

        st.error(
            "No processed audio contexts were found."
        )

        st.stop()

    # ------------------------------------------------------
    # SELECT CONTEXT
    # ------------------------------------------------------

    selected_context = st.selectbox(
        "Select an audio context",
        context_ids,
        index=0
    )

    context = load_context(
        selected_context
    )

    if context is None:

        st.error(
            "Unable to load the selected context."
        )

        st.stop()

    # ------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------

    audio_path = (
        CONTEXT_AUDIO_DIR
        / f"{selected_context}.wav"
    )

    col1, col2 = st.columns(
        [1.5, 1]
    )

    with col1:

        st.subheader(
            "Audio Sample"
        )

        if audio_path.exists():

            with open(
                audio_path,
                "rb"
            ) as audio_file:

                audio_bytes = audio_file.read()

            st.audio(
                audio_bytes,
                format="audio/wav"
            )

        else:

            st.warning(
                "Audio file not found."
            )

    with col2:

        st.subheader(
            "Context Summary"
        )

        duration = context.get(
            "duration",
            0
        )

        event_count = context.get(
            "num_events",
            len(
                context.get(
                    "events",
                    []
                )
            )
        )

        unique_count = context.get(
            "num_unique_event_types",
            0
        )

        st.metric(
            "Duration",
            f"{duration:.2f} s"
        )

        st.metric(
            "Events",
            event_count
        )

        st.metric(
            "Unique sound types",
            unique_count
        )

    st.divider()

    # ------------------------------------------------------
    # EVENT SUMMARY
    # ------------------------------------------------------

    st.subheader(
        "Detected Sound Sequence"
    )

    sequence = context.get(
        "event_sequence",
        []
    )

    if sequence:

        st.code(
            "  →  ".join(sequence),
            language="text"
        )

    # ------------------------------------------------------
    # TIMELINE
    # ------------------------------------------------------

    with st.expander(
        "View event timeline",
        expanded=True
    ):

        display_event_timeline(
            context
        )

    st.divider()

    # ------------------------------------------------------
    # QUESTION
    # ------------------------------------------------------

    st.subheader(
        "Ask a Question"
    )

    example_questions = [
        "What sounds are present?",
        "How many bird sound events occur?",
        "How many cat sound events occur?",
        "What happens after the first bird?",
        "What happens before the second dog?",
        "Which sound occurs first?",
        "Which sound occurs last?",
        "What is the sequence of sounds?",
        "How long does the first bird sound last?",
        "Why might a dog sound occur?"
    ]

    example = st.selectbox(
        "Example question",
        ["Custom question"] + example_questions
    )

    if example == "Custom question":

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What happens after the first bird?"
            )
        )

    else:

        question = example

        st.text_input(
            "Selected question",
            value=question,
            disabled=True
        )

    ask = st.button(
        "🔎 Analyze Audio",
        type="primary",
        width="stretch"
    )

    # ------------------------------------------------------
    # ANSWER
    # ------------------------------------------------------

    if ask:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Analyzing audio context..."
            ):

                try:

                    result = pipeline.answer(
                        question,
                        selected_context
                    )

                    question_type = result.get(
                        "question_type",
                        "unknown"
                    )

                    answer = result.get(
                        "answer",
                        "No answer generated."
                    )

                    retrieved = result.get(
                        "retrieved_context",
                        {}
                    )

                except Exception as error:

                    st.error(
                        f"Pipeline error: {error}"
                    )

                    st.stop()

            st.divider()

            # --------------------------------------------------
            # RESULT HEADER
            # --------------------------------------------------

            left, right = st.columns(
                [1, 3]
            )

            with left:

                st.metric(
                    "Question Type",
                    question_type.upper()
                )

            with right:

                st.markdown(
                    "### Answer"
                )

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # --------------------------------------------------
            # RETRIEVED CONTEXT
            # --------------------------------------------------

            with st.expander(
                "View retrieved context"
            ):

                st.json(
                    retrieved
                )


# ==========================================================
# CONTEXT EXPLORER
# ==========================================================

elif page == "Context Explorer":

    st.header(
        "🔍 Context Explorer"
    )

    context_ids = get_context_ids()

    if not context_ids:

        st.warning(
            "No context files available."
        )

        st.stop()

    selected_context = st.selectbox(
        "Select context",
        context_ids
    )

    context = load_context(
        selected_context
    )

    if context is None:

        st.error(
            "Context could not be loaded."
        )

        st.stop()

    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Context",
            selected_context
        )

    with col2:

        st.metric(
            "Duration",
            f"{context.get('duration', 0):.2f}s"
        )

    with col3:

        st.metric(
            "Events",
            context.get(
                "num_events",
                0
            )
        )

    with col4:

        st.metric(
            "Unique Types",
            context.get(
                "num_unique_event_types",
                0
            )
        )

    st.divider()

    # ------------------------------------------------------
    # EVENT COUNTS
    # ------------------------------------------------------

    st.subheader(
        "Event Counts"
    )

    counts = context.get(
        "event_counts",
        {}
    )

    if counts:

        count_df = pd.DataFrame(
            {
                "Sound": list(
                    counts.keys()
                ),
                "Count": list(
                    counts.values()
                )
            }
        )

        st.bar_chart(
            count_df.set_index(
                "Sound"
            )
        )

    # ------------------------------------------------------
    # EVENT TIMELINE
    # ------------------------------------------------------

    st.subheader(
        "Temporal Event Timeline"
    )

    display_event_timeline(
        context
    )

    # ------------------------------------------------------
    # STRUCTURED JSON
    # ------------------------------------------------------

    with st.expander(
        "View raw Context JSON"
    ):

        st.json(
            context
        )


# ==========================================================
# EVALUATION
# ==========================================================

elif page == "Evaluation":

    st.header(
        "📊 Test Set Evaluation"
    )

    evaluation = load_evaluation()

    if evaluation is None:

        st.warning(
            "evaluation_results.json was not found."
        )

        st.stop()

    dataset = evaluation.get(
        "dataset",
        {}
    )

    overall = evaluation.get(
        "overall",
        {}
    )

    breakdown = evaluation.get(
        "breakdown_by_question_type",
        {}
    )

    # ------------------------------------------------------
    # OVERALL METRICS
    # ------------------------------------------------------

    st.subheader(
        "Overall Results"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Question Classification",
            percent(
                overall.get(
                    "question_classification_accuracy",
                    0
                )
            )
        )

    with c2:

        st.metric(
            "Structured Fact Accuracy",
            percent(
                overall.get(
                    "structured_fact_accuracy",
                    0
                )
            )
        )

    with c3:

        st.metric(
            "Answer Token F1",
            f"{overall.get('answer_token_f1', 0):.4f}"
        )

    with c4:

        st.metric(
            "Test Questions",
            dataset.get(
                "test_questions",
                0
            )
        )

    st.divider()

    # ------------------------------------------------------
    # DATASET SUMMARY
    # ------------------------------------------------------

    st.subheader(
        "Evaluation Dataset"
    )

    d1, d2 = st.columns(2)

    with d1:

        st.metric(
            "Test Contexts",
            dataset.get(
                "test_contexts",
                0
            )
        )

    with d2:

        st.metric(
            "Test Questions",
            dataset.get(
                "test_questions",
                0
            )
        )

    # ------------------------------------------------------
    # QUESTION TYPE BREAKDOWN
    # ------------------------------------------------------

    st.subheader(
        "Performance by Question Type"
    )

    rows = []

    for question_type, metrics in breakdown.items():

        rows.append(
            {
                "Question Type": question_type.capitalize(),
                "Questions": metrics.get(
                    "total_questions",
                    0
                ),
                "Structured Fact Accuracy": (
                    metrics.get(
                        "structured_fact_accuracy",
                        0
                    ) * 100
                ),
                "Token F1": metrics.get(
                    "answer_token_f1",
                    0
                )
            }
        )

    if rows:

        breakdown_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            breakdown_df.style.format(
                {
                    "Structured Fact Accuracy":
                        "{:.2f}%",
                    "Token F1":
                        "{:.4f}"
                }
            ),
            width="stretch",
            hide_index=True
        )

        chart_df = breakdown_df[
            [
                "Question Type",
                "Structured Fact Accuracy"
            ]
        ].set_index(
            "Question Type"
        )

        st.bar_chart(
            chart_df
        )

    # ------------------------------------------------------
    # SYSTEM INFORMATION
    # ------------------------------------------------------

    st.divider()

    st.subheader(
        "System Information"
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write(
            "**Audio contexts:** 100"
        )

        st.write(
            "**Dataset clips:** 610"
        )

        st.write(
            "**Supported sounds:** bird, cat, dog"
        )

    with info_col2:

        st.write(
            "**QA types:** perceptual, counting, temporal, "
            "duration, sequence, causal"
        )

        st.write(
            "**Test contexts:** 15"
        )

        st.write(
            "**Pipeline errors:** 0"
        )