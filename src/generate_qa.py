import json
import random
from pathlib import Path
from collections import Counter


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_annotations"
)

TRAIN_FILE = PROJECT_ROOT / "data" / "train.json"
VALIDATION_FILE = PROJECT_ROOT / "data" / "validation.json"
TEST_FILE = PROJECT_ROOT / "data" / "test.json"


# ==========================================================
# SETTINGS
# ==========================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================================================
# ORDINAL HELPER
# ==========================================================

def ordinal(number):
    """Convert 1,2,3... to first, second, third..."""

    values = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth"
    }

    return values.get(number, f"{number}th")


def build_event_reference(events, index):
    """
    Create an unambiguous human-readable reference
    for an event.

    Example:
        bird occurring once  -> bird
        repeated bird       -> first bird / second bird
    """

    event_name = events[index]["name"]

    counts = Counter(
        event["name"]
        for event in events
    )

    if counts[event_name] == 1:
        return event_name

    occurrence = 0

    for i in range(index + 1):

        if events[i]["name"] == event_name:
            occurrence += 1

    return f"{ordinal(occurrence)} {event_name}"


# ==========================================================
# GENERATE QUESTIONS FOR ONE CONTEXT
# ==========================================================

def generate_questions(annotation):

    events = annotation.get("events", [])

    if not events:
        return []

    questions = []

    # ------------------------------------------------------
    # Count unique sound classes
    # ------------------------------------------------------

    counts = Counter(
        event["name"]
        for event in events
    )

    # ======================================================
    # PERCEPTUAL
    # ======================================================

    unique_names = list(dict.fromkeys(
        event["name"]
        for event in events
    ))

    if len(unique_names) == 1:

        perceptual_answer = (
            f"The audio contains "
            f"{unique_names[0]} sounds."
        )

    elif len(unique_names) == 2:

        perceptual_answer = (
            f"The audio contains "
            f"{unique_names[0]} and "
            f"{unique_names[1]} sounds."
        )

    else:

        perceptual_answer = (
            "The audio contains "
            + ", ".join(unique_names[:-1])
            + " and "
            + unique_names[-1]
            + " sounds."
        )

    questions.append({
        "question": "What sounds are present?",
        "type": "perceptual",
        "answer": perceptual_answer
    })

    # ======================================================
    # COUNTING
    # ======================================================

    for name in sorted(counts):

        count = counts[name]

        questions.append({
            "question": (
                f"How many {name} sound events occur?"
            ),
            "type": "counting",
            "answer": (
                f"There are {count} "
                f"{name} sound event(s)."
            )
        })

    # ======================================================
    # TEMPORAL - BEFORE / AFTER
    # ======================================================

    for index in range(len(events) - 1):

        current = build_event_reference(
            events,
            index
        )

        next_event = events[index + 1]["name"]

        # ----------------------------------------------
        # AFTER
        # ----------------------------------------------

        questions.append({
            "question": (
                f"What happens after the {current}?"
            ),
            "type": "temporal",
            "answer": (
                f"The {next_event} occurs after "
                f"the {current}."
            )
        })

        # ----------------------------------------------
        # BEFORE
        # ----------------------------------------------

        next_reference = build_event_reference(
            events,
            index + 1
        )

        previous_event = events[index]["name"]

        questions.append({
            "question": (
                f"What happens before the "
                f"{next_reference}?"
            ),
            "type": "temporal",
            "answer": (
                f"The {previous_event} occurs before "
                f"the {next_reference}."
            )
        })

    # ======================================================
    # FIRST EVENT
    # ======================================================

    first_event = events[0]["name"]

    questions.append({
        "question": "Which sound occurs first?",
        "type": "temporal",
        "answer": (
            f"The {first_event} occurs first."
        )
    })

    # ======================================================
    # LAST EVENT
    # ======================================================

    last_event = events[-1]["name"]

    questions.append({
        "question": "Which sound occurs last?",
        "type": "temporal",
        "answer": (
            f"The {last_event} occurs last."
        )
    })

    # ======================================================
    # SEQUENCE
    # ======================================================

    sequence = [
        event["name"]
        for event in events
    ]

    questions.append({
        "question": "What is the sequence of sounds?",
        "type": "sequence",
        "answer": (
            "The sequence is: "
            + " -> ".join(sequence)
            + "."
        )
    })

    # ======================================================
    # DURATION
    # ======================================================

    for index, event in enumerate(events):

        duration = round(
            event["end"] - event["start"],
            2
        )

        reference = build_event_reference(
            events,
            index
        )

        if counts[event["name"]] == 1:

            question = (
                f"How long does the "
                f"{reference} sound last?"
            )

            answer = (
                f"The {reference} sound lasts "
                f"approximately {duration:.2f} seconds."
            )

        else:

            question = (
                f"How long does the "
                f"{reference} sound last?"
            )

            answer = (
                f"The {reference} sound lasts "
                f"approximately {duration:.2f} seconds."
            )

        questions.append({
            "question": question,
            "type": "duration",
            "answer": answer
        })

    # ======================================================
    # CAUSAL
    # ======================================================

    for name in sorted(counts):

        questions.append({
            "question": (
                f"Why might a {name} sound occur?"
            ),
            "type": "causal",
            "answer": (
                "The audio alone cannot establish the "
                f"cause. The {name} sound may simply "
                "indicate the presence of that sound "
                "source in the recorded environment."
            )
        })

    return questions


# ==========================================================
# BUILD DATASET
# ==========================================================

def main():

    print("=" * 60)
    print("AUDIO CONTEXT LAYER")
    print("TEMPORALLY UNAMBIGUOUS QA GENERATION")
    print("=" * 60)

    annotation_files = sorted(
        ANNOTATIONS_DIR.glob("*.json")
    )

    print(
        f"Annotation files found: "
        f"{len(annotation_files)}"
    )

    records = []

    for index, annotation_file in enumerate(
        annotation_files,
        start=1
    ):

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as file:

            annotation = json.load(file)

        audio_id = annotation_file.stem

        record = {
            "audio_id": audio_id,
            "audio_file": (
                f"context_audio/{audio_id}.wav"
            ),
            "duration": annotation.get(
                "duration",
                0
            ),
            "environment": "synthetic_multi_event",
            "split": annotation.get("split"),
            "events": annotation.get(
                "events",
                []
            ),
            "questions": generate_questions(
                annotation
            )
        }

        records.append(record)

        if index % 10 == 0:
            print(
                f"Processed {index}/"
                f"{len(annotation_files)}"
            )

    # ======================================================
    # PRESERVE SOURCE-DISJOINT CONTEXT SPLITS
    # ======================================================

    total = len(records)

    train_data = [
        record
        for record in records
        if record.get("split") == "train"
    ]

    validation_data = [
        record
        for record in records
        if record.get("split") == "validation"
    ]

    test_data = [
        record
        for record in records
        if record.get("split") == "test"
    ]

    # ------------------------------------------------------
    # Validate splits
    # ------------------------------------------------------

    missing_split = [
        record["audio_id"]
        for record in records
        if record.get("split")
        not in {"train", "validation", "test"}
    ]

    if missing_split:

        raise ValueError(
            "Some contexts are missing a valid split: "
            + ", ".join(missing_split)
        )

    # ======================================================
    # SAVE DATASETS
    # ======================================================

    datasets = [
        (TRAIN_FILE, train_data),
        (VALIDATION_FILE, validation_data),
        (TEST_FILE, test_data)
    ]

    for output_file, data in datasets:

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=2
            )

    # ======================================================
    # STATISTICS
    # ======================================================

    type_counter = Counter()

    total_questions = 0

    for record in records:

        for question in record["questions"]:

            type_counter[
                question["type"]
            ] += 1

            total_questions += 1

    print()

    print(
        f"Audio contexts: {total}"
    )

    print(
        f"Training contexts: "
        f"{len(train_data)}"
    )

    print(
        f"Validation contexts: "
        f"{len(validation_data)}"
    )

    print(
        f"Test contexts: "
        f"{len(test_data)}"
    )

    print(
        f"Total QA pairs: "
        f"{total_questions}"
    )

    print()

    print(
        "Question type distribution:"
    )

    for question_type in sorted(
        type_counter
    ):

        print(
            f"  {question_type}: "
            f"{type_counter[question_type]}"
        )

    print()

    print(
        "Dataset generation completed successfully."
    )


if __name__ == "__main__":
    main()