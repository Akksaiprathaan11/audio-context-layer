from pathlib import Path
import json

import joblib
import librosa
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "audio_event_classifier.joblib"
)

CONTEXT_AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_audio"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SAMPLE_RATE = 16000

N_MFCC = 40

SEGMENT_DURATION = 1.0


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("AUDIO EVENT PREDICTOR")
print("=" * 60)

print(
    "\nLoading model..."
)

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    audio
):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std
        ]
    )

    return features.reshape(
        1,
        -1
    )


# ============================================================
# PREDICT ONE AUDIO FILE
# ============================================================

def predict_audio(
    audio_path
):

    y, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    segment_samples = int(
        SEGMENT_DURATION
        * SAMPLE_RATE
    )

    events = []

    total_samples = len(y)

    segment_start = 0.0

    sample_start = 0


    while sample_start < total_samples:

        sample_end = min(
            sample_start
            +
            segment_samples,
            total_samples
        )

        segment = y[
            sample_start:sample_end
        ]

        # Skip extremely short segments
        if len(segment) < int(
            SAMPLE_RATE * 0.25
        ):
            break

        # Pad short final segment
        if len(segment) < segment_samples:

            segment = np.pad(
                segment,
                (
                    0,
                    segment_samples
                    - len(segment)
                )
            )

        features = extract_features(
            segment
        )

        prediction = model.predict(
            features
        )[0]

        probabilities = (
            model.predict_proba(
                features
            )[0]
        )

        class_names = (
            model.classes_
        )

        confidence = float(
            probabilities[
                list(class_names)
                .index(prediction)
            ]
        )

        segment_end = min(
            segment_start
            +
            SEGMENT_DURATION,
            len(y) / SAMPLE_RATE
        )

        events.append(
            {
                "name":
                    str(prediction),

                "start":
                    round(
                        segment_start,
                        3
                    ),

                "end":
                    round(
                        segment_end,
                        3
                    ),

                "confidence":
                    round(
                        confidence,
                        4
                    )
            }
        )

        sample_start = (
            sample_end
        )

        segment_start = (
            sample_end
            /
            SAMPLE_RATE
        )


    return {
        "audio_file":
            audio_path.name,

        "duration":
            round(
                len(y)
                /
                SAMPLE_RATE,
                3
            ),

        "events":
            events
    }


# ============================================================
# PROCESS ALL CONTEXT AUDIO
# ============================================================

audio_files = sorted(
    CONTEXT_AUDIO_DIR.glob(
        "*.wav"
    )
)

print(
    f"\nContext audio files found: "
    f"{len(audio_files)}"
)


if not audio_files:

    raise FileNotFoundError(
        "No context WAV files found."
    )


for index, audio_file in enumerate(
    audio_files,
    start=1
):

    result = predict_audio(
        audio_file
    )

    output_file = (
        PREDICTION_DIR
        /
        f"{audio_file.stem}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        f"[{index}/{len(audio_files)}] "
        f"{audio_file.name}"
    )


print(
    "\nPrediction completed successfully."
)

print(
    f"Predictions saved to:"
)

print(
    PREDICTION_DIR
)