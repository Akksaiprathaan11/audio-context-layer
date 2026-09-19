from pathlib import Path
import json
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "predictions"
)

CONTEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "contexts"
)

CONTEXT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(prediction):

    events = prediction.get(
        "events",
        []
    )

    # Sort chronologically
    events = sorted(
        events,
        key=lambda event: event["start"]
    )

    # --------------------------------------------------------
    # EVENT COUNTS
    # --------------------------------------------------------

    event_counts = Counter(
        event["name"]
        for event in events
    )

    event_counts = dict(
        event_counts
    )

    # --------------------------------------------------------
    # EVENT SEQUENCE
    # --------------------------------------------------------

    event_sequence = [
        event["name"]
        for event in events
    ]

    # --------------------------------------------------------
    # UNIQUE EVENT TYPES
    # --------------------------------------------------------

    unique_events = list(
        dict.fromkeys(
            event_sequence
        )
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context = {

        "audio_file":
            prediction["audio_file"],

        "duration":
            prediction["duration"],

        "events":
            events,

        "event_counts":
            event_counts,

        "event_sequence":
            event_sequence,

        "unique_events":
            unique_events,

        "num_events":
            len(events),

        "num_unique_event_types":
            len(unique_events)
    }

    return context


# ============================================================
# PROCESS ALL PREDICTIONS
# ============================================================

def process_all_predictions():

    prediction_files = sorted(
        PREDICTION_DIR.glob(
            "context_*.json"
        )
    )

    print("=" * 60)
    print("AUDIO CONTEXT LAYER")
    print("=" * 60)

    print(
        f"\nPrediction files found: "
        f"{len(prediction_files)}"
    )

    if not prediction_files:

        raise FileNotFoundError(
            "No prediction files found."
        )

    for index, prediction_file in enumerate(
        prediction_files,
        start=1
    ):

        with open(
            prediction_file,
            "r",
            encoding="utf-8"
        ) as f:

            prediction = json.load(f)

        context = build_context(
            prediction
        )

        output_file = (
            CONTEXT_DIR
            /
            prediction_file.name
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                context,
                f,
                indent=2
            )

        if index % 10 == 0:

            print(
                f"Processed "
                f"{index}/"
                f"{len(prediction_files)}"
            )

    print(
        "\nContext building completed successfully."
    )

    print(
        f"Contexts saved to:\n"
        f"{CONTEXT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_all_predictions()