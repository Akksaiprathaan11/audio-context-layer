from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIR = ROOT / "data" / "processed" / "context_audio"
ANNOTATION_DIR = ROOT / "data" / "processed" / "context_annotations"

VALID_TYPES = {
    "perceptual",
    "counting",
    "temporal",
    "sequence",
    "duration",
    "causal"
}

errors = []

annotation_files = sorted(
    ANNOTATION_DIR.glob("context_*.json")
)

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

for file in annotation_files:

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_id = data["audio_id"]
    duration = float(data["duration"])

    audio_file = AUDIO_DIR / f"{audio_id}.wav"

    if not audio_file.exists():
        errors.append(
            f"Missing audio: {audio_file}"
        )

    events = data.get("events", [])

    if not events:
        errors.append(
            f"No events: {audio_id}"
        )

    for event in events:

        start = float(event["start"])
        end = float(event["end"])

        if start < 0:
            errors.append(
                f"Negative start: {audio_id}"
            )

        if end <= start:
            errors.append(
                f"Invalid event duration: {audio_id}"
            )

        if end > duration + 0.01:
            errors.append(
                f"Event exceeds duration: {audio_id}"
            )

    qa_file = (
        ROOT
        / "data"
        / "processed"
        / "qa"
        / f"{audio_id}.json"
    )

    if not qa_file.exists():
        errors.append(
            f"Missing QA file: {audio_id}"
        )
        continue

    with open(
        qa_file,
        "r",
        encoding="utf-8"
    ) as f:
        qa_data = json.load(f)

    for question in qa_data.get(
        "questions",
        []
    ):

        if not question.get("question"):
            errors.append(
                f"Missing question: {audio_id}"
            )

        if not question.get("answer"):
            errors.append(
                f"Missing answer: {audio_id}"
            )

        qtype = question.get("type")

        if qtype not in VALID_TYPES:
            errors.append(
                f"Invalid question type "
                f"{qtype}: {audio_id}"
            )


print(
    f"\nContexts checked: "
    f"{len(annotation_files)}"
)

if errors:

    print(
        f"\nFound {len(errors)} errors:"
    )

    for error in errors[:50]:
        print(" -", error)

else:

    print(
        "\n✓ DATASET VALIDATION PASSED"
    )