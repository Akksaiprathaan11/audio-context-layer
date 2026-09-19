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

SPLIT_FILE = PROJECT_ROOT / "data" / "source_splits.json"

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

RANDOM_SEED = 42

CONTEXTS_PER_SPLIT = {
    "train": 70,
    "validation": 15,
    "test": 15
}

MIN_EVENTS = 3
MAX_EVENTS = 5


# ==========================================================
# AUDIO NORMALIZATION
# ==========================================================

def load_and_normalize_audio(audio_path):
    """
    Load audio as mono 16 kHz and normalize it to exactly
    one second.
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
# LOAD SOURCE SPLITS
# ==========================================================

def load_source_splits():

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)["splits"]


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 65)
    print("AUDIO CONTEXT LAYER")
    print("SOURCE-DISJOINT CONTEXT GENERATION")
    print("=" * 65)

    random.seed(RANDOM_SEED)

    splits = load_source_splits()

    # ------------------------------------------------------
    # Validate split manifest
    # ------------------------------------------------------

    print()
    print("Source split summary:")

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        print()
        print(
            f"{split_name.upper()}"
        )

        for class_name in sorted(
            splits[split_name]
        ):

            print(
                f"  {class_name}: "
                f"{len(splits[split_name][class_name])}"
            )

    # ------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------

    CONTEXT_AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CONTEXT_ANNOTATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove old generated context files
    for wav_file in CONTEXT_AUDIO_DIR.glob("*.wav"):
        wav_file.unlink()

    for json_file in CONTEXT_ANNOTATIONS_DIR.glob("*.json"):
        json_file.unlink()

    # ------------------------------------------------------
    # Global context number
    # ------------------------------------------------------

    context_number = 1

    total_created = 0

    # ------------------------------------------------------
    # Generate train / validation / test separately
    # ------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        target_count = CONTEXTS_PER_SPLIT[
            split_name
        ]

        print()
        print(
            f"Generating {split_name} contexts: "
            f"{target_count}"
        )

        # Build available source pool for this split
        source_pool = {}

        for class_name in sorted(
            splits[split_name]
        ):

            source_pool[class_name] = []

            for relative_path in splits[
                split_name
            ][class_name]:

                full_path = (
                    PROJECT_ROOT
                    / relative_path
                )

                if full_path.exists():
                    source_pool[class_name].append(
                        (
                            relative_path,
                            full_path
                        )
                    )

        class_names = sorted(
            source_pool.keys()
        )

        for local_index in range(
            1,
            target_count + 1
        ):

            number_of_events = random.randint(
                MIN_EVENTS,
                MAX_EVENTS
            )

            audio_parts = []

            events = []

            current_time = 0.0

            for event_index in range(
                number_of_events
            ):

                class_name = random.choice(
                    class_names
                )

                relative_path, source_path = (
                    random.choice(
                        source_pool[class_name]
                    )
                )

                audio = load_and_normalize_audio(
                    source_path
                )

                audio_parts.append(
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

                    "source_file": relative_path
                })

                current_time += EVENT_DURATION

            # --------------------------------------------------
            # Combine audio
            # --------------------------------------------------

            context_audio = np.concatenate(
                audio_parts
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
            # Save audio
            # --------------------------------------------------

            sf.write(
                wav_path,
                context_audio,
                SAMPLE_RATE
            )

            # --------------------------------------------------
            # Save annotation
            # --------------------------------------------------

            annotation = {

                "audio_id":
                    context_id,

                "audio_file":
                    f"context_audio/{context_id}.wav",

                "duration":
                    round(
                        current_time,
                        3
                    ),

                "environment":
                    "synthetic_multi_event",

                "split":
                    split_name,

                "events":
                    events
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

            total_created += 1

            if (
                local_index == 1
                or local_index % 10 == 0
                or local_index == target_count
            ):

                print(
                    f"  [{local_index}/{target_count}] "
                    f"{context_id}"
                )

            context_number += 1

    # ======================================================
    # VERIFY
    # ======================================================

    annotation_files = list(
        CONTEXT_ANNOTATIONS_DIR.glob(
            "*.json"
        )
    )

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    for annotation_file in annotation_files:

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as file:

            annotation = json.load(file)

        split_name = annotation.get(
            "split"
        )

        if split_name in split_counts:
            split_counts[split_name] += 1

    # ======================================================
    # SUMMARY
    # ======================================================

    print()
    print("=" * 65)
    print("SOURCE-DISJOINT CONTEXT GENERATION COMPLETED")
    print("=" * 65)

    print(
        f"Train contexts:      "
        f"{split_counts['train']}"
    )

    print(
        f"Validation contexts: "
        f"{split_counts['validation']}"
    )

    print(
        f"Test contexts:       "
        f"{split_counts['test']}"
    )

    print(
        f"Total contexts:      "
        f"{total_created}"
    )

    print()
    print(
        "Every context contains only source files "
        "from its assigned split."
    )

    print()
    print(
        "Context audio saved to:"
    )

    print(
        CONTEXT_AUDIO_DIR
    )

    print()
    print(
        "Context annotations saved to:"
    )

    print(
        CONTEXT_ANNOTATIONS_DIR
    )


if __name__ == "__main__":
    main()