from pathlib import Path
import json

def load_annotated_events(annotation_path: str | Path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", []), data

def detect_events_from_annotations(annotation_path: str | Path):
    events, _ = load_annotated_events(annotation_path)
    return sorted(events, key=lambda x: float(x.get("start", 0.0)))

def summarize_events(events):
    counts = {}
    for event in events:
        name = event["name"]
        counts[name] = counts.get(name, 0) + 1
    return counts
