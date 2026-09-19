import json
import re
from pathlib import Path
from collections import defaultdict


from pipeline import AudioContextPipeline


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_FILE = PROJECT_ROOT / "data" / "test.json"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "evaluation_results.json"


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize_text(text):
    """
    Normalize text for secondary exact-match comparison.
    """

    text = str(text).lower()

    text = text.replace("→", " ")
    text = text.replace("->", " ")

    text = re.sub(r"[^a-z0-9.\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# TOKEN F1
# ==========================================================

def token_f1(predicted, reference):
    """
    Calculate token-level F1.
    """

    pred_tokens = normalize_text(
        predicted
    ).split()

    ref_tokens = normalize_text(
        reference
    ).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}

    for token in pred_tokens:
        pred_counts[token] = (
            pred_counts.get(token, 0) + 1
        )

    ref_counts = {}

    for token in ref_tokens:
        ref_counts[token] = (
            ref_counts.get(token, 0) + 1
        )

    overlap = 0

    for token in pred_counts:

        if token in ref_counts:

            overlap += min(
                pred_counts[token],
                ref_counts[token]
            )

    if overlap == 0:
        return 0.0

    precision = (
        overlap / len(pred_tokens)
    )

    recall = (
        overlap / len(ref_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ==========================================================
# EVENT DETECTION
# ==========================================================

def detect_event(question):
    """
    Detect bird, cat, or dog from a question.
    """

    question = question.lower()

    for event in [
        "bird",
        "cat",
        "dog"
    ]:

        if re.search(
            rf"\b{event}\b",
            question
        ):
            return event

    return None


# ==========================================================
# OCCURRENCE DETECTION
# ==========================================================

def extract_occurrence(question):
    """
    Extract ordinal occurrence.

    Examples:
        first bird  -> 1
        second bird -> 2
        third dog   -> 3
    """

    question = question.lower()

    ordinal_map = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10
    }

    for word, number in ordinal_map.items():

        if re.search(
            rf"\b{word}\b",
            question
        ):

            return number

    return None


# ==========================================================
# GOLD SEQUENCE
# ==========================================================

def get_gold_sequence(record):

    return [
        event["name"]
        for event in record.get(
            "events",
            []
        )
    ]


# ==========================================================
# GOLD COUNTS
# ==========================================================

def get_gold_counts(record):

    counts = defaultdict(int)

    for event in record.get(
        "events",
        []
    ):

        counts[event["name"]] += 1

    return dict(counts)


# ==========================================================
# FIND GOLD EVENT INDEX
# ==========================================================

def find_gold_event_index(
    gold_events,
    event_name,
    occurrence=None
):
    """
    Find the exact event index.

    occurrence=None:
        first occurrence

    occurrence=1:
        first occurrence

    occurrence=2:
        second occurrence
    """

    indexes = [
        index
        for index, event in enumerate(
            gold_events
        )
        if event["name"] == event_name
    ]

    if not indexes:
        return None

    if occurrence is None:

        return indexes[0]

    if occurrence > len(indexes):
        return None

    return indexes[
        occurrence - 1
    ]


# ==========================================================
# STRUCTURED FACT EVALUATION
# ==========================================================

def evaluate_structured_fact(
    record,
    question_item,
    pipeline_result
):
    """
    Evaluate the underlying audio/context fact.

    This does not depend on exact wording of the
    generated natural-language answer.
    """

    question = question_item["question"]

    question_type = question_item["type"]

    q = question.lower().strip()

    retrieved = pipeline_result[
        "retrieved_context"
    ]

    gold_events = record.get(
        "events",
        []
    )

    gold_sequence = get_gold_sequence(
        record
    )

    gold_counts = get_gold_counts(
        record
    )

    target_event = detect_event(
        question
    )

    occurrence = extract_occurrence(
        question
    )

    # ======================================================
    # PERCEPTUAL
    # ======================================================

    if question_type == "perceptual":

        predicted = set(
            retrieved.get(
                "unique_events",
                []
            )
        )

        gold = set(
            gold_sequence
        )

        return predicted == gold

    # ======================================================
    # COUNTING
    # ======================================================

    if question_type == "counting":

        if target_event is None:
            return False

        predicted_count = retrieved.get(
            "count"
        )

        gold_count = gold_counts.get(
            target_event,
            0
        )

        return (
            predicted_count
            == gold_count
        )

    # ======================================================
    # SEQUENCE
    # ======================================================

    if question_type == "sequence":

        predicted_sequence = retrieved.get(
            "event_sequence",
            []
        )

        return (
            predicted_sequence
            == gold_sequence
        )

    # ======================================================
    # DURATION
    # ======================================================

    if question_type == "duration":

        if target_event is None:
            return False

        retrieved_events = retrieved.get(
            "events",
            []
        )

        target_index = find_gold_event_index(
            gold_events,
            target_event,
            occurrence
        )

        if target_index is None:
            return False

        gold_duration = (
            gold_events[target_index]["end"]
            - gold_events[target_index]["start"]
        )

        # For ordinal duration questions we expect
        # exactly the referenced event.
        if occurrence is not None:

            if len(retrieved_events) != 1:
                return False

            predicted_duration = (
                retrieved_events[0].get(
                    "duration"
                )
            )

            if predicted_duration is None:
                return False

            return abs(
                predicted_duration
                - gold_duration
            ) <= 0.05

        # For a non-ordinal duration query,
        # match any correct occurrence.
        for event in retrieved_events:

            predicted_duration = event.get(
                "duration"
            )

            if predicted_duration is None:
                continue

            if abs(
                predicted_duration
                - gold_duration
            ) <= 0.05:

                return True

        return False

    # ======================================================
    # TEMPORAL
    # ======================================================

    if question_type == "temporal":

        # --------------------------------------------------
        # FIRST EVENT
        # --------------------------------------------------
        # IMPORTANT:
        # This must only match the standalone first-event
        # question, not "after the first bird".

        if (
            "which sound occurs first" in q
            or q == "what happened first?"
            or "first event" in q
        ):

            predicted = retrieved.get(
                "first_event"
            )

            if predicted is None:
                return False

            if not gold_events:
                return False

            return (
                predicted.get("name")
                == gold_events[0].get("name")
            )

        # --------------------------------------------------
        # LAST EVENT
        # --------------------------------------------------

        if (
            "which sound occurs last" in q
            or q == "what happened last?"
            or "last event" in q
        ):

            predicted = retrieved.get(
                "last_event"
            )

            if predicted is None:
                return False

            if not gold_events:
                return False

            return (
                predicted.get("name")
                == gold_events[-1].get("name")
            )

        # --------------------------------------------------
        # BEFORE / AFTER
        # --------------------------------------------------

        if target_event is None:
            return False

        target_index = find_gold_event_index(
            gold_events,
            target_event,
            occurrence
        )

        if target_index is None:
            return False

        # --------------------------------------------------
        # AFTER
        # --------------------------------------------------

        if "after" in q:

            expected_index = (
                target_index + 1
            )

            predicted = retrieved.get(
                "next_event"
            )

            # Target is the final event.
            if expected_index >= len(
                gold_events
            ):

                return predicted is None

            # A next event must exist.
            if predicted is None:
                return False

            expected_name = gold_events[
                expected_index
            ]["name"]

            return (
                predicted.get("name")
                == expected_name
            )

        # --------------------------------------------------
        # BEFORE
        # --------------------------------------------------

        if "before" in q:

            expected_index = (
                target_index - 1
            )

            predicted = retrieved.get(
                "previous_event"
            )

            # Target is the first event.
            if expected_index < 0:

                return predicted is None

            # A previous event must exist.
            if predicted is None:
                return False

            expected_name = gold_events[
                expected_index
            ]["name"]

            return (
                predicted.get("name")
                == expected_name
            )

        return False

    # ======================================================
    # CAUSAL
    # ======================================================

    if question_type == "causal":

        answer = pipeline_result[
            "answer"
        ].lower()

        return (
            "cannot establish" in answer
            or "cannot determine" in answer
            or "cannot" in answer
        )

    return False


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 70)
    print("AUDIO CONTEXT LAYER - TEST SET EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------
    # LOAD TEST SET
    # ------------------------------------------------------

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        test_data = json.load(file)

    print(
        f"Test contexts: {len(test_data)}"
    )

    # ------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------

    pipeline = AudioContextPipeline()

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    total_questions = 0

    classifier_correct = 0

    structured_correct = 0

    exact_correct = 0

    total_token_f1 = 0.0

    errors = 0

    by_type = defaultdict(
        lambda: {
            "total": 0,
            "classifier_correct": 0,
            "structured_correct": 0,
            "exact_match": 0,
            "token_f1": 0.0
        }
    )

    detailed_results = []

    # ------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------

    for record in test_data:

        context_id = record["audio_id"]

        questions = record.get(
            "questions",
            []
        )

        for question_item in questions:

            question = question_item[
                "question"
            ]

            gold_type = question_item[
                "type"
            ]

            gold_answer = question_item[
                "answer"
            ]

            total_questions += 1

            # ----------------------------------------------
            # RUN PIPELINE
            # ----------------------------------------------

            try:

                result = pipeline.answer(
                    question,
                    context_id
                )

            except Exception as error:

                errors += 1

                print()
                print(
                    f"ERROR: {context_id} | "
                    f"{question}"
                )

                print(
                    f"Reason: {error}"
                )

                continue

            # ----------------------------------------------
            # QUESTION CLASSIFICATION
            # ----------------------------------------------

            predicted_type = result[
                "question_type"
            ]

            classifier_ok = (
                predicted_type == gold_type
            )

            if classifier_ok:
                classifier_correct += 1

            # ----------------------------------------------
            # STRUCTURED FACT
            # ----------------------------------------------

            structured_ok = (
                evaluate_structured_fact(
                    record,
                    question_item,
                    result
                )
            )

            if structured_ok:
                structured_correct += 1

            # ----------------------------------------------
            # ANSWER METRICS
            # ----------------------------------------------

            predicted_answer = result[
                "answer"
            ]

            exact_ok = (
                normalize_text(
                    predicted_answer
                )
                ==
                normalize_text(
                    gold_answer
                )
            )

            if exact_ok:
                exact_correct += 1

            current_f1 = token_f1(
                predicted_answer,
                gold_answer
            )

            total_token_f1 += current_f1

            # ----------------------------------------------
            # TYPE METRICS
            # ----------------------------------------------

            metrics = by_type[
                gold_type
            ]

            metrics["total"] += 1

            if classifier_ok:
                metrics[
                    "classifier_correct"
                ] += 1

            if structured_ok:
                metrics[
                    "structured_correct"
                ] += 1

            if exact_ok:
                metrics[
                    "exact_match"
                ] += 1

            metrics[
                "token_f1"
            ] += current_f1

            # ----------------------------------------------
            # DETAILED RESULT
            # ----------------------------------------------

            detailed_results.append({

                "context_id":
                    context_id,

                "question":
                    question,

                "gold_type":
                    gold_type,

                "predicted_type":
                    predicted_type,

                "classification_correct":
                    classifier_ok,

                "structured_correct":
                    structured_ok,

                "gold_answer":
                    gold_answer,

                "predicted_answer":
                    predicted_answer,

                "exact_match":
                    exact_ok,

                "token_f1":
                    round(
                        current_f1,
                        4
                    )
            })

    # ======================================================
    # OVERALL METRICS
    # ======================================================

    classifier_accuracy = (
        classifier_correct / total_questions
        if total_questions
        else 0
    )

    structured_accuracy = (
        structured_correct / total_questions
        if total_questions
        else 0
    )

    exact_accuracy = (
        exact_correct / total_questions
        if total_questions
        else 0
    )

    average_token_f1 = (
        total_token_f1 / total_questions
        if total_questions
        else 0
    )

    # ======================================================
    # BREAKDOWN
    # ======================================================

    breakdown = {}

    for question_type, metrics in sorted(
        by_type.items()
    ):

        total = metrics["total"]

        breakdown[question_type] = {

            "total_questions":
                total,

            "question_classification_accuracy":
                round(
                    metrics[
                        "classifier_correct"
                    ] / total,
                    4
                ) if total else 0,

            "structured_fact_accuracy":
                round(
                    metrics[
                        "structured_correct"
                    ] / total,
                    4
                ) if total else 0,

            "answer_exact_match":
                round(
                    metrics[
                        "exact_match"
                    ] / total,
                    4
                ) if total else 0,

            "answer_token_f1":
                round(
                    metrics[
                        "token_f1"
                    ] / total,
                    4
                ) if total else 0
        }

    # ======================================================
    # RESULTS OBJECT
    # ======================================================

    results = {

        "dataset": {

            "test_contexts":
                len(test_data),

            "test_questions":
                total_questions
        },

        "overall": {

            "question_classification_accuracy":
                round(
                    classifier_accuracy,
                    4
                ),

            "structured_fact_accuracy":
                round(
                    structured_accuracy,
                    4
                ),

            "answer_exact_match":
                round(
                    exact_accuracy,
                    4
                ),

            "answer_token_f1":
                round(
                    average_token_f1,
                    4
                )
        },

        "breakdown_by_question_type":
            breakdown,

        "detailed_results":
            detailed_results
    }

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    # ======================================================
    # PRINT OVERALL
    # ======================================================

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"Question classification accuracy: "
        f"{classifier_accuracy * 100:.2f}%"
    )

    print(
        f"Structured fact accuracy: "
        f"{structured_accuracy * 100:.2f}%"
    )

    print(
        f"Answer exact match: "
        f"{exact_accuracy * 100:.2f}%"
    )

    print(
        f"Answer token F1: "
        f"{average_token_f1:.4f}"
    )

    print(
        f"Pipeline errors: "
        f"{errors}"
    )

    # ======================================================
    # PRINT BREAKDOWN
    # ======================================================

    print()
    print("=" * 70)
    print("BREAKDOWN BY QUESTION TYPE")
    print("=" * 70)

    for question_type, metrics in breakdown.items():

        print()
        print(
            question_type.upper()
        )

        print(
            f"Questions: "
            f"{metrics['total_questions']}"
        )

        print(
            f"Classification Accuracy: "
            f"{metrics['question_classification_accuracy'] * 100:.2f}%"
        )

        print(
            f"Structured Fact Accuracy: "
            f"{metrics['structured_fact_accuracy'] * 100:.2f}%"
        )

        print(
            f"Exact Match: "
            f"{metrics['answer_exact_match'] * 100:.2f}%"
        )

        print(
            f"Token F1: "
            f"{metrics['answer_token_f1']:.4f}"
        )

    print()
    print("=" * 70)
    print("Evaluation completed successfully.")
    print("=" * 70)

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()