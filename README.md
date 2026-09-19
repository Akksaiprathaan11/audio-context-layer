# Audio Context Layer — Audio Question Answering PoC

An end-to-end proof of concept that converts audio into a structured temporal context and answers natural-language questions grounded in that context.

## Architecture

Audio -> preprocessing -> audio event detection -> temporal context -> question classification -> context retrieval -> answer generation

The starter implementation uses a lightweight, deterministic event layer based on configurable annotations/heuristics. This keeps the PoC runnable without requiring a large pretrained checkpoint. A Gemini API key can optionally be used for natural-language answer generation.

## Question types

- Perceptual
- Counting
- Temporal
- Causal/reasoning
- Duration
- Sequence

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the CLI with an annotated JSON example:

```powershell
python src/pipeline.py --audio data/raw/audio_001.wav --question "How many times does the dog bark?" --annotations data/processed/audio_001.json
```

Run the Streamlit demo:

```powershell
streamlit run app/streamlit_app.py
```

Run evaluation:

```powershell
python src/evaluation.py
```

## Dataset format

Each processed audio JSON contains:

```json
{
  "audio_id": "audio_001",
  "duration": 8.0,
  "environment": "outdoor",
  "events": [
    {"name": "dog_bark", "start": 1.0, "end": 1.6, "confidence": 0.95}
  ],
  "questions": [
    {
      "question": "How many times does the dog bark?",
      "type": "counting",
      "answer": "The dog barks 2 times."
    }
  ]
}
```

Place WAV files in `data/raw/` and matching annotations in `data/processed/`.

## Important

The supplied code is a complete runnable PoC scaffold, but the audio event detector is intentionally lightweight. For a research-grade submission, replace `src/audio_events.py` with a pretrained AudioSet/YAMNet/PANNs-style detector and evaluate its predictions on the held-out test set.

Do not put API keys directly in source code. If using Gemini, set `GEMINI_API_KEY` in the environment.

## Suggested report contents

1. Problem formulation
2. Research study
3. Dataset description and construction methodology
4. Method and design decisions
5. Experimental setup
6. Results
7. Loss curves if a trainable component is used
8. Error analysis
9. Observations and limitations
