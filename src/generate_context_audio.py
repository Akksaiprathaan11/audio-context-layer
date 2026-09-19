import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

CONTEXT_AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_audio"
)

CONTEXT_ANNOTATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_annotations"
)


# ==========================================================
# SETTINGS
# ==========================================================

SAMPLE_RATE = 16000

EVENT_DURATION = 1.0

EVENT_SAMPLES = int(
    SAMPLE_RATE * EVENT_DURATION
)

NUM_CONTEXTS = 100

MIN_EVENTS = 3
MAX_EVENTS = 5

RANDOM_SEED = 42


# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

CONTEXT_AUDIO_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CONTEXT_ANNOTATIONS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD AUDIO
# ==========================================================

def load_and_normalize_audio(audio_path):
    """
    Load audio as mono 16 kHz and force it to exactly
    one second.

    Longer files are truncated.
    Shorter files are zero-padded.
    """

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if len(audio) > EVENT_SAMPLES:

        audio = audio[
            :EVENT_SAMPLES
        ]

    elif len(audio) < EVENT_SAMPLES:

        padding = (
            EVENT_SAMPLES
            - len(audio)
        )

        audio = np.pad(
            audio,
            (0, padding)
        )

    return audio


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("AUDIO CONTEXT LAYER")
    print("FIXED-DURATION CONTEXT GENERATION")
    print("=" * 60)

    random.seed(
        RANDOM_SEED
    )

    # ------------------------------------------------------
    # Discover classes
    # ------------------------------------------------------

    classes = {}

    for class_dir in sorted(
        RAW_DIR.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        wav_files = sorted(
            class_dir.glob("*.wav")
        )

        if wav_files:

            classes[
                class_dir.name
            ] = wav_files

    if not classes:

        raise RuntimeError(
            f"No WAV files found in {RAW_DIR}"
        )

    print()
    print("Available classes:")

    for class_name, files in classes.items():

        print(
            f"{class_name}: "
            f"{len(files)} files"
        )

    print()

    # ------------------------------------------------------
    # Remove old generated contexts
    # ------------------------------------------------------

    for wav_file in (
        CONTEXT_AUDIO_DIR.glob("*.wav")
    ):

        wav_file.unlink()

    for annotation_file in (
        CONTEXT_ANNOTATIONS_DIR.glob("*.json")
    ):

        annotation_file.unlink()

    # ------------------------------------------------------
    # Generate contexts
    # ------------------------------------------------------

    generated = 0

    class_names = sorted(
        classes.keys()
    )

    for context_number in range(
        1,
        NUM_CONTEXTS + 1
    ):

        number_of_events = random.randint(
            MIN_EVENTS,
            MAX_EVENTS
        )

        context_audio_parts = []

        events = []

        current_time = 0.0

        for event_index in range(
            number_of_events
        ):

            # Randomly choose sound class
            class_name = random.choice(
                class_names
            )

            # Random source file
            source_file = random.choice(
                classes[class_name]
            )

            # Load and normalize to exactly 1 second
            audio = load_and_normalize_audio(
                source_file
            )

            context_audio_parts.append(
                audio
            )

            start_time = round(
                current_time,
                3
            )

            end_time = round(
                current_time
                + EVENT_DURATION,
                3
            )

            events.append({
                "name": class_name,
                "start": start_time,
                "end": end_time,
                "source_file": str(
                    source_file
                )
            })

            current_time += (
                EVENT_DURATION
            )

        # --------------------------------------------------
        # Combine events
        # --------------------------------------------------

        context_audio = np.concatenate(
            context_audio_parts
        )

        context_id = (
            f"context_{context_number:04d}"
        )

        wav_path = (
            CONTEXT_AUDIO_DIR
            / f"{context_id}.wav"
        )

        annotation_path = (
            CONTEXT_ANNOTATIONS_DIR
            / f"{context_id}.json"
        )

        # --------------------------------------------------
        # Save WAV
        # --------------------------------------------------

        sf.write(
            wav_path,
            context_audio,
            SAMPLE_RATE
        )

        # --------------------------------------------------
        # Build annotation
        # --------------------------------------------------

        annotation = {
            "audio_id": context_id,
            "audio_file": (
                f"context_audio/"
                f"{context_id}.wav"
            ),
            "duration": round(
                current_time,
                3
            ),
            "environment": (
                "synthetic_multi_event"
            ),
            "events": events
        }

        with open(
            annotation_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                annotation,
                file,
                indent=2
            )

        generated += 1

        if (
            context_number == 1
            or context_number % 10 == 0
        ):

            print(
                f"Created {context_id} "
                f"with {number_of_events} events"
            )

    # ======================================================
    # COMPLETE
    # ======================================================

    print()

    print(
        f"Generated contexts: "
        f"{generated}"
    )

    print(
        f"Each event duration: "
        f"{EVENT_DURATION:.1f} second"
    )

    print(
        "Multi-event audio generation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()