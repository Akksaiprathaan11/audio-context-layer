import re


QUESTION_TYPES = [
    "perceptual",
    "counting",
    "temporal",
    "sequence",
    "duration",
    "causal"
]


def classify_question(question):
    """
    Classify a natural-language question into one of the
    supported Audio Context Layer question types.
    """

    q = question.lower().strip()

    # Causal / reasoning questions
    causal_patterns = [
        r"\bwhy\b",
        r"\bwhat caused\b",
        r"\bwhat is the reason\b",
        r"\bhow did\b.*\bhappen\b",
        r"\bcause\b"
    ]

    for pattern in causal_patterns:
        if re.search(pattern, q):
            return "causal"

    # Duration questions
    duration_patterns = [
        r"\bhow long\b",
        r"\bduration\b",
        r"\bhow many seconds\b",
        r"\bhow much time\b"
    ]

    for pattern in duration_patterns:
        if re.search(pattern, q):
            return "duration"

    # Counting questions
    counting_patterns = [
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bcount\b",
        r"\bhow much\b"
    ]

    for pattern in counting_patterns:
        if re.search(pattern, q):
            return "counting"

    # Temporal questions
    temporal_patterns = [
        r"\bbefore\b",
        r"\bafter\b",
        r"\bwhen\b",
        r"\bfirst\b",
        r"\blast\b",
        r"\bearlier\b",
        r"\blater\b"
    ]

    for pattern in temporal_patterns:
        if re.search(pattern, q):
            return "temporal"

    # Sequence questions
    sequence_patterns = [
        r"\border\b",
        r"\bsequence\b",
        r"\bwhat happened first\b",
        r"\bwhat happened next\b",
        r"\bwhat happened last\b"
    ]

    for pattern in sequence_patterns:
        if re.search(pattern, q):
            return "sequence"

    # Default: perceptual
    return "perceptual"


if __name__ == "__main__":

    test_questions = [
        "What sounds are present in the audio?",
        "How many bird sounds are there?",
        "What happened before the cat sound?",
        "What happened after the dog sound?",
        "What is the order of the sounds?",
        "How long did the bird sound last?",
        "Why did the dog sound occur?"
    ]

    print("=" * 60)
    print("QUESTION CLASSIFIER TEST")
    print("=" * 60)

    for question in test_questions:
        question_type = classify_question(question)
        print(f"Question: {question}")
        print(f"Type:     {question_type}")
        print("-" * 60)