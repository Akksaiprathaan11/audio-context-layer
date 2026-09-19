from pathlib import Path
import json
import random
import librosa


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = PROJECT_ROOT / "data" / "train.json"
VALIDATION_FILE = PROJECT_ROOT / "data" / "validation.json"
TEST_FILE = PROJECT_ROOT / "data" / "test.json"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# CREATE DIRECTORIES
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND AUDIO
# ============================================================

audio_files = sorted(
    RAW_DIR.rglob("*.wav")
)

print("=" * 60)
print("AUDIO CONTEXT LAYER - DATASET PREPARATION")
print("=" * 60)

print(
    f"\nAudio files found: {len(audio_files)}"
)

if not audio_files:
    raise FileNotFoundError(
        "No WAV files found inside data/raw/"
    )


# ============================================================
# CREATE DATASET
# ============================================================

dataset = []

class_counts = {}


for index, audio_path in enumerate(
    audio_files,
    start=1
):

    try:

        # --------------------------------------------
        # CLASS FROM FOLDER NAME
        # --------------------------------------------

        sound_class = audio_path.parent.name.lower()

        # --------------------------------------------
        # AUDIO INFORMATION
        # --------------------------------------------

        duration = librosa.get_duration(
            path=str(audio_path)
        )

        relative_path = audio_path.relative_to(
            RAW_DIR
        )

        audio_id = f"audio_{index:04d}"

        # --------------------------------------------
        # EVENT
        # --------------------------------------------

        event = {
            "name": sound_class,
            "start": 0.0,
            "end": round(
                float(duration),
                3
            ),
            "confidence": 1.0
        }

        # --------------------------------------------
        # QUESTIONS
        # --------------------------------------------

        questions = [

            {
                "question": "What sound is present?",
                "type": "perceptual",
                "answer":
                    f"A {sound_class} sound is present."
            },

            {
                "question":
                    f"Is a {sound_class} sound audible?",
                "type": "perceptual",
                "answer":
                    f"Yes, a {sound_class} sound is audible."
            },

            {
                "question":
                    "What is the main sound in the recording?",
                "type": "perceptual",
                "answer":
                    f"The main sound is a {sound_class}."
            },

            {
                "question":
                    "What type of sound does the audio contain?",
                "type": "perceptual",
                "answer":
                    f"The audio contains a {sound_class} sound."
            }

        ]

        # --------------------------------------------
        # RECORD
        # --------------------------------------------

        item = {

            "audio_id": audio_id,

            "audio_file":
                str(relative_path).replace(
                    "\\",
                    "/"
                ),

            "duration":
                round(
                    float(duration),
                    3
                ),

            "environment":
                "unknown",

            "events": [
                event
            ],

            "questions":
                questions
        }

        dataset.append(item)

        # --------------------------------------------
        # CLASS STATISTICS
        # --------------------------------------------

        class_counts[sound_class] = (
            class_counts.get(
                sound_class,
                0
            ) + 1
        )

        # --------------------------------------------
        # SAVE INDIVIDUAL JSON
        # --------------------------------------------

        output_file = (
            PROCESSED_DIR /
            f"{audio_id}.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                item,
                f,
                indent=2
            )

        if index % 50 == 0:

            print(
                f"Processed "
                f"{index}/{len(audio_files)}"
            )

    except Exception as e:

        print(
            f"ERROR: "
            f"{audio_path.name} -> {e}"
        )


# ============================================================
# SHUFFLE
# ============================================================

random.seed(
    RANDOM_SEED
)

random.shuffle(
    dataset
)


# ============================================================
# SPLIT
# ============================================================

total = len(dataset)

train_end = int(
    total * TRAIN_RATIO
)

validation_end = (
    train_end
    +
    int(total * VALIDATION_RATIO)
)

train_data = dataset[
    :train_end
]

validation_data = dataset[
    train_end:validation_end
]

test_data = dataset[
    validation_end:
]


# ============================================================
# SAVE
# ============================================================

def save_json(
    data,
    path
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


save_json(
    train_data,
    TRAIN_FILE
)

save_json(
    validation_data,
    VALIDATION_FILE
)

save_json(
    test_data,
    TEST_FILE
)


# ============================================================
# STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DATASET CREATED")
print("=" * 60)

print(
    f"\nTotal audio files: {total}"
)

print(
    f"Training: {len(train_data)}"
)

print(
    f"Validation: {len(validation_data)}"
)

print(
    f"Test: {len(test_data)}"
)

print("\nSound classes:")

for name, count in sorted(
    class_counts.items()
):

    print(
        f"  {name}: {count}"
    )

print(
    "\nPerceptual QA pairs:"
)

print(
    f"  {total * 4}"
)

print(
    "\nDataset preparation completed."
)