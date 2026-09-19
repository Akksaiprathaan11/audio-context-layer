import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import json
import tempfile
import streamlit as st
from pipeline import run_pipeline

st.set_page_config(page_title="Audio Context Layer", page_icon="🎧", layout="centered")
st.title("🎧 Audio Context Layer")
st.caption("Audio Question Answering PoC")

audio = st.file_uploader("Upload a WAV/MP3 audio file", type=["wav", "mp3", "flac"])
question = st.text_input("Ask a question about the audio")

st.info(
    "For the supplied PoC, upload an audio file and provide a matching annotation JSON "
    "through the CLI for deterministic event-grounded answers. The UI can still demonstrate "
    "the interaction flow, while production deployment should connect an event detector."
)

if audio and question and st.button("Ask"):
    suffix = Path(audio.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio.read())
        audio_path = f.name

    result = run_pipeline(audio_path, question)
    st.subheader("Answer")
    st.write(result["answer"])
    st.subheader("Question Type")
    st.write(result["question_type"])
    st.subheader("Context")
    st.json(result["context"])
