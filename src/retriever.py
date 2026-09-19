import json
import re
from pathlib import Path


class AudioContextRetriever:
    """
    Retrieves relevant information from the Audio Context Layer.
    """

    def __init__(self, contexts_dir="data/processed/contexts"):
        self.contexts_dir = Path(contexts_dir)

    # ==========================================================
    # LOAD CONTEXT
    # ==========================================================

    def load_context(self, context_id):
        """Load a context JSON file."""

        if not context_id.endswith(".json"):
            context_id += ".json"

        path = self.contexts_dir / context_id

        if not path.exists():
            raise FileNotFoundError(
                f"Context file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    # ==========================================================
    # ORDINAL HELPER
    # ==========================================================

    def ordinal(self, number):
        """Convert numbers to ordinal words."""

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

        return values.get(
            number,
            f"{number}th"
        )

    # ==========================================================
    # DETECT EVENT
    # ==========================================================

    def detect_event(self, question):
        """Detect bird, cat, or dog in the question."""

        question = question.lower()

        for event in ["bird", "cat", "dog"]:

            if re.search(
                rf"\b{event}\b",
                question
            ):
                return event

        return None

    # ==========================================================
    # DETECT EVENT + OCCURRENCE
    # ==========================================================

    def detect_event_reference(self, question):
        """
        Detect event name and optional occurrence number.

        Examples:
            first bird  -> ("bird", 1)
            second bird -> ("bird", 2)
            third dog   -> ("dog", 3)
            dog         -> ("dog", None)
        """

        question = question.lower()

        ordinal_map = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5
        }

        for event in ["bird", "cat", "dog"]:

            for ordinal_word, occurrence in ordinal_map.items():

                pattern = (
                    rf"\b{ordinal_word}\s+"
                    rf"{event}\b"
                )

                if re.search(
                    pattern,
                    question
                ):
                    return event, occurrence

        return self.detect_event(question), None

    # ==========================================================
    # NORMALIZE EVENTS
    # ==========================================================

    def normalize_events(self, context):
        """
        Normalize the event structure.

        Original format:
        {
            "name": "bird",
            "start": 0.0,
            "end": 1.0,
            "confidence": 0.84
        }
        """

        normalized = []

        for index, event in enumerate(
            context.get("events", [])
        ):

            start = event.get("start")
            end = event.get("end")

            duration = None

            if start is not None and end is not None:

                duration = round(
                    end - start,
                    3
                )

            normalized.append({
                "index": index,
                "name": event.get("name"),
                "start": start,
                "end": end,
                "duration": duration,
                "confidence": event.get(
                    "confidence"
                )
            })

        return normalized

    # ==========================================================
    # FIND EVENT INDEX
    # ==========================================================

    def find_event_index(
        self,
        events,
        event_name,
        occurrence=None
    ):
        """
        Find the index of an event.

        occurrence=None:
            use first occurrence.

        occurrence=1:
            first occurrence.

        occurrence=2:
            second occurrence.
        """

        indexes = [
            i
            for i, item in enumerate(events)
            if item["name"] == event_name
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
    # RETRIEVE
    # ==========================================================

    def retrieve(self, question, context):

        q = question.lower().strip()

        event = self.detect_event(question)

        events = self.normalize_events(context)

        result = {
            "question": question,
            "event": event,
            "relevant_context": {}
        }

        # ======================================================
        # COUNTING
        # ======================================================

        if (
            "how many" in q
            or "number of" in q
            or "count" in q
        ):

            counts = context.get(
                "event_counts",
                {}
            )

            if event:

                result["relevant_context"] = {
                    "event": event,
                    "count": counts.get(
                        event,
                        0
                    )
                }

            else:

                result["relevant_context"] = {
                    "event_counts": counts
                }

        # ======================================================
        # DURATION
        # ======================================================

        elif (
            "how long" in q
            or "duration" in q
            or "how many seconds" in q
        ):

            event_name, occurrence = (
                self.detect_event_reference(
                    question
                )
            )

            matching = []

            if event_name:

                if occurrence is not None:

                    target_index = (
                        self.find_event_index(
                            events,
                            event_name,
                            occurrence
                        )
                    )

                    if target_index is not None:

                        matching.append(
                            events[target_index]
                        )

                else:

                    matching = [
                        item
                        for item in events
                        if item["name"] == event_name
                    ]

            else:

                matching = events

            result["relevant_context"] = {
                "events": matching
            }

        # ======================================================
        # BEFORE
        # ======================================================

        elif "before" in q:

            event_name, occurrence = (
                self.detect_event_reference(
                    question
                )
            )

            if event_name:

                target_index = (
                    self.find_event_index(
                        events,
                        event_name,
                        occurrence
                    )
                )

                if target_index is None:

                    reference = event_name

                    if occurrence is not None:
                        reference = (
                            f"{self.ordinal(occurrence)} "
                            f"{event_name}"
                        )

                    result["relevant_context"] = {
                        "target_event": event_name,
                        "target_occurrence": occurrence,
                        "previous_event": None,
                        "message": (
                            f"No {reference} event "
                            "was detected."
                        )
                    }

                elif target_index > 0:

                    result["relevant_context"] = {
                        "target_event": event_name,
                        "target_occurrence": occurrence,
                        "target_index": target_index,
                        "previous_event": events[
                            target_index - 1
                        ]
                    }

                else:

                    result["relevant_context"] = {
                        "target_event": event_name,
                        "target_occurrence": occurrence,
                        "target_index": target_index,
                        "previous_event": None,
                        "message": (
                            "The referenced event "
                            "is the first event."
                        )
                    }

        # ======================================================
        # AFTER
        # ======================================================

        elif "after" in q:

            event_name, occurrence = (
                self.detect_event_reference(
                    question
                )
            )

            if event_name:

                target_index = (
                    self.find_event_index(
                        events,
                        event_name,
                        occurrence
                    )
                )

                if target_index is None:

                    reference = event_name

                    if occurrence is not None:
                        reference = (
                            f"{self.ordinal(occurrence)} "
                            f"{event_name}"
                        )

                    result["relevant_context"] = {
                        "target_event": event_name,
                        "target_occurrence": occurrence,
                        "next_event": None,
                        "message": (
                            f"No {reference} event "
                            "was detected."
                        )
                    }

                elif target_index < len(events) - 1:

                    result["relevant_context"] = {
                        "target_event": event_name,
                        "target_occurrence": occurrence,
                        "target_index": target_index,
                        "next_event": events[
                            target_index + 1
                        ]
                    }

                else:

                    result["relevant_context"] = {
                        "target_event": event_name,
                        "target_occurrence": occurrence,
                        "target_index": target_index,
                        "next_event": None,
                        "message": (
                            "The referenced event "
                            "is the last event."
                        )
                    }

        # ======================================================
        # FIRST EVENT
        # ======================================================

        elif (
            "what happened first" in q
            or "which sound occurs first" in q
            or "first event" in q
        ):

            if events:

                result["relevant_context"] = {
                    "first_event": events[0]
                }

        # ======================================================
        # LAST EVENT
        # ======================================================

        elif (
            "what happened last" in q
            or "which sound occurs last" in q
            or "last event" in q
        ):

            if events:

                result["relevant_context"] = {
                    "last_event": events[-1]
                }

        # ======================================================
        # SEQUENCE / ORDER
        # ======================================================

        elif (
            "order" in q
            or "sequence" in q
            or "what happened next" in q
        ):

            result["relevant_context"] = {
                "event_sequence": context.get(
                    "event_sequence",
                    []
                )
            }

        # ======================================================
        # CAUSAL
        # ======================================================

        elif (
            "why" in q
            or "cause" in q
            or "reason" in q
        ):

            result["relevant_context"] = {
                "event": event,
                "causal_reasoning": (
                    "The audio context can identify the "
                    "sound event, but audio alone cannot "
                    "establish its real-world cause."
                )
            }

        # ======================================================
        # PERCEPTUAL
        # ======================================================

        else:

            result["relevant_context"] = {
                "unique_events": context.get(
                    "unique_events",
                    []
                ),
                "num_events": context.get(
                    "num_events",
                    0
                ),
                "num_unique_event_types": context.get(
                    "num_unique_event_types",
                    0
                )
            }

        return result


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AUDIO CONTEXT RETRIEVER TEST")
    print("=" * 60)

    retriever = AudioContextRetriever()

    context_id = "context_0001"

    try:

        context = retriever.load_context(
            context_id
        )

        print(
            f"Loaded: {context_id}"
        )

        print()

        questions = [

            "What sounds are present in the audio?",

            "How many bird sounds are there?",

            "How many dog sounds are there?",

            "How long does the first bird sound last?",

            "How long does the second bird sound last?",

            "What happened before the dog sound?",

            "What happens after the first bird?",

            "What happens after the second bird?",

            "What happens before the second bird?",

            "What happens before the third bird?",

            "What is the order of the sounds?",

            "What happened first?",

            "What happened last?",

            "Why did the dog sound occur?"
        ]

        for question in questions:

            print(
                f"Question: {question}"
            )

            output = retriever.retrieve(
                question,
                context
            )

            print("Retrieved:")

            print(
                json.dumps(
                    output[
                        "relevant_context"
                    ],
                    indent=2
                )
            )

            print("-" * 60)

        print(
            "Retriever test completed successfully."
        )

    except Exception as error:

        print(
            f"ERROR: {error}"
        )