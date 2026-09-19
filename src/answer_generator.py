class AudioAnswerGenerator:
    """
    Converts retrieved Audio Context information
    into natural-language answers.
    """

    # ==================================================
    # ORDINAL HELPER
    # ==================================================

    def ordinal(self, number):
        """Convert occurrence number to ordinal word."""

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

    # ==================================================
    # EVENT REFERENCE
    # ==================================================

    def build_target_reference(self, retrieved_context):
        """
        Build an event reference such as:

        bird
        first bird
        second bird
        third cat
        """

        target = retrieved_context.get(
            "target_event"
        )

        occurrence = retrieved_context.get(
            "target_occurrence"
        )

        if not target:
            return "the target event"

        if occurrence is not None:

            return (
                f"{self.ordinal(occurrence)} "
                f"{target}"
            )

        return target

    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    def generate(self, question, retrieved_context):

        # ==================================================
        # PERCEPTUAL
        # ==================================================

        if "unique_events" in retrieved_context:

            events = retrieved_context.get(
                "unique_events",
                []
            )

            num_events = retrieved_context.get(
                "num_events",
                0
            )

            if not events:

                return (
                    "No recognizable sound events "
                    "were detected."
                )

            event_text = ", ".join(events)

            return (
                f"The audio contains {num_events} "
                f"sound events, involving: {event_text}."
            )

        # ==================================================
        # COUNTING
        # ==================================================

        if (
            "event" in retrieved_context
            and "count" in retrieved_context
        ):

            event = retrieved_context["event"]

            count = retrieved_context["count"]

            if count == 0:

                return (
                    f"No {event} sounds were detected "
                    "in the audio."
                )

            if count == 1:

                return (
                    f"There is 1 {event} sound "
                    "in the audio."
                )

            return (
                f"There are {count} {event} sounds "
                "in the audio."
            )

        # ==================================================
        # DURATION
        # ==================================================

        if "events" in retrieved_context:

            events = retrieved_context["events"]

            if not events:

                return (
                    "No matching sound event was detected "
                    "in the audio."
                )

            event_name = events[0]["name"]

            if len(events) == 1:

                duration = events[0]["duration"]

                occurrence = retrieved_context.get(
                    "target_occurrence"
                )

                if occurrence is not None:

                    reference = (
                        f"{self.ordinal(occurrence)} "
                        f"{event_name}"
                    )

                    return (
                        f"The {reference} sound lasts "
                        f"approximately "
                        f"{duration:.3f} seconds."
                    )

                return (
                    f"The {event_name} sound lasts "
                    f"approximately "
                    f"{duration:.3f} seconds."
                )

            total_duration = sum(
                item["duration"]
                for item in events
            )

            return (
                f"There are {len(events)} "
                f"{event_name} events with a combined "
                f"duration of approximately "
                f"{total_duration:.3f} seconds."
            )

        # ==================================================
        # BEFORE
        # ==================================================

        if "previous_event" in retrieved_context:

            previous = retrieved_context[
                "previous_event"
            ]

            target_reference = (
                self.build_target_reference(
                    retrieved_context
                )
            )

            if previous is None:

                return (
                    f"There was no event before the "
                    f"{target_reference} sound because "
                    "it was the first event."
                )

            return (
                f"A {previous['name']} sound occurred "
                f"before the {target_reference} sound."
            )

        # ==================================================
        # AFTER
        # ==================================================

        if "next_event" in retrieved_context:

            next_event = retrieved_context[
                "next_event"
            ]

            target_reference = (
                self.build_target_reference(
                    retrieved_context
                )
            )

            if next_event is None:

                return (
                    f"There was no event after the "
                    f"{target_reference} sound because "
                    "it was the last event."
                )

            return (
                f"A {next_event['name']} sound occurred "
                f"after the {target_reference} sound."
            )

        # ==================================================
        # FIRST EVENT
        # ==================================================

        if "first_event" in retrieved_context:

            event = retrieved_context[
                "first_event"
            ]

            return (
                f"The first sound event was "
                f"{event['name']}, occurring from "
                f"{event['start']:.3f} to "
                f"{event['end']:.3f} seconds."
            )

        # ==================================================
        # LAST EVENT
        # ==================================================

        if "last_event" in retrieved_context:

            event = retrieved_context[
                "last_event"
            ]

            return (
                f"The last sound event was "
                f"{event['name']}, occurring from "
                f"{event['start']:.3f} to "
                f"{event['end']:.3f} seconds."
            )

        # ==================================================
        # SEQUENCE
        # ==================================================

        if "event_sequence" in retrieved_context:

            sequence = retrieved_context[
                "event_sequence"
            ]

            if not sequence:

                return (
                    "No sound sequence was detected."
                )

            formatted_sequence = " → ".join(
                sequence
            )

            return (
                f"The sound sequence is: "
                f"{formatted_sequence}."
            )

        # ==================================================
        # CAUSAL
        # ==================================================

        if "causal_reasoning" in retrieved_context:

            return retrieved_context[
                "causal_reasoning"
            ]

        # ==================================================
        # FALLBACK
        # ==================================================

        return (
            "I could not find enough relevant audio "
            "context to answer that question."
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AUDIO ANSWER GENERATOR TEST")
    print("=" * 60)

    generator = AudioAnswerGenerator()

    test_cases = [

        # --------------------------------------------------
        # PERCEPTUAL
        # --------------------------------------------------

        (
            "What sounds are present?",
            {
                "unique_events": [
                    "bird",
                    "dog"
                ],
                "num_events": 5,
                "num_unique_event_types": 2
            }
        ),

        # --------------------------------------------------
        # COUNTING
        # --------------------------------------------------

        (
            "How many bird sounds are there?",
            {
                "event": "bird",
                "count": 4
            }
        ),

        # --------------------------------------------------
        # DURATION
        # --------------------------------------------------

        (
            "How long does the first bird sound last?",
            {
                "target_event": "bird",
                "target_occurrence": 1,
                "events": [
                    {
                        "name": "bird",
                        "duration": 1.0
                    }
                ]
            }
        ),

        (
            "How long does the second bird sound last?",
            {
                "target_event": "bird",
                "target_occurrence": 2,
                "events": [
                    {
                        "name": "bird",
                        "duration": 1.0
                    }
                ]
            }
        ),

        # --------------------------------------------------
        # BEFORE
        # --------------------------------------------------

        (
            "What happened before the first dog?",
            {
                "target_event": "dog",
                "target_occurrence": 1,
                "previous_event": {
                    "name": "cat"
                }
            }
        ),

        (
            "What happened before the second bird?",
            {
                "target_event": "bird",
                "target_occurrence": 2,
                "previous_event": {
                    "name": "dog"
                }
            }
        ),

        # --------------------------------------------------
        # AFTER
        # --------------------------------------------------

        (
            "What happens after the first bird?",
            {
                "target_event": "bird",
                "target_occurrence": 1,
                "next_event": {
                    "name": "dog"
                }
            }
        ),

        (
            "What happens after the second bird?",
            {
                "target_event": "bird",
                "target_occurrence": 2,
                "next_event": {
                    "name": "cat"
                }
            }
        ),

        # --------------------------------------------------
        # SEQUENCE
        # --------------------------------------------------

        (
            "What is the order of the sounds?",
            {
                "event_sequence": [
                    "bird",
                    "dog",
                    "bird",
                    "bird",
                    "bird"
                ]
            }
        ),

        # --------------------------------------------------
        # FIRST
        # --------------------------------------------------

        (
            "What happened first?",
            {
                "first_event": {
                    "name": "bird",
                    "start": 0.0,
                    "end": 1.0
                }
            }
        ),

        # --------------------------------------------------
        # LAST
        # --------------------------------------------------

        (
            "What happened last?",
            {
                "last_event": {
                    "name": "bird",
                    "start": 4.0,
                    "end": 4.896
                }
            }
        ),

        # --------------------------------------------------
        # CAUSAL
        # --------------------------------------------------

        (
            "Why did the dog sound occur?",
            {
                "event": "dog",
                "causal_reasoning": (
                    "The audio context can identify the "
                    "sound event, but audio alone cannot "
                    "establish its real-world cause."
                )
            }
        )
    ]
    for question, context in test_cases:

        print()
        print(
            f"Question: {question}"
        )

        answer = generator.generate(
            question,
            context
        )

        print(
            f"Answer: {answer}"
        )

        print("-" * 60)

    print(
        "\nAnswer generator test completed successfully."
    )