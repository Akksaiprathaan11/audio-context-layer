import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from question_classifier import classify_question
from context_builder import build_context
from answer_generator import generate_answer

def test_question_classifier():
    assert classify_question("How many times does the dog bark?") == "counting"
    assert classify_question("What happens after the horn?") == "temporal"
    assert classify_question("Why might the horn occur?") == "causal"

def test_count_answer():
    events = [
        {"name": "dog_bark", "start": 1.0, "end": 1.5},
        {"name": "dog_bark", "start": 2.0, "end": 2.5},
    ]
    context = build_context(events)
    answer = generate_answer("How many times does the dog bark?", "counting", context)
    assert "2" in answer
