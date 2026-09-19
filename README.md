# Audio Context Layer

A proof-of-concept Audio Question Answering system that converts audio into a structured temporal context layer and answers natural-language questions grounded in detected sound events.

## 1. Project Overview

The project is designed to answer questions such as:

- What sounds are present?
- How many times does a sound occur?
- What happens before or after a particular event?
- What is the temporal sequence of sounds?
- How long does an event last?
- Why might a sound occur?

The system uses a lightweight audio-event classification pipeline followed by structured context construction, question classification, context retrieval, and answer generation.

## 2. Architecture

```text
Audio Clips
    |
    v
1-second Audio Segmentation
    |
    v
MFCC Feature Extraction
    |
    v
Random Forest Audio Classifier
    |
    v
Predicted Sound Events
    |
    v
Audio Context Layer
- event names
- start/end timestamps
- duration
- confidence
- event counts
- event sequence
    |
    v
Question Classifier
    |
    v
Context Retriever
    |
    v
Answer Generator
    |
    v
Natural-language Answer
```

## 3. Dataset

The project uses 610 original WAV clips from three sound classes:

| Class | Clips |
|---|---:|
| Bird | 193 |
| Cat | 207 |
| Dog | 210 |
| **Total** | **610** |

A source-level split is used to reduce leakage across train/validation/test:

| Split | Source clips |
|---|---:|
| Train | 426 |
| Validation | 90 |
| Test | 94 |

The project additionally constructs 100 synthetic multi-event contexts:

| Split | Contexts |
|---|---:|
| Train | 70 |
| Validation | 15 |
| Test | 15 |
| **Total** | **100** |

Each context contains a sequence of 1-second sound events. The context annotations store event labels and temporal boundaries.

## 4. Generated QA Dataset

The final QA generator creates 1,878 question-answer pairs.

| Question type | QA pairs |
|---|---:|
| Perceptual | 100 |
| Counting | 236 |
| Temporal | 804 |
| Duration | 402 |
| Sequence | 100 |
| Causal | 236 |
| **Total** | **1,878** |

Temporal questions use occurrence-aware references such as `first bird`, `second bird`, etc., so repeated events can be distinguished.

## 5. Project Structure

```text
audio-context-layer/
├── app/
│   └── app.py
├── data/
│   ├── original/
│   ├── processed/
│   │   ├── context_audio/
│   │   ├── context_annotations/
│   │   ├── contexts/
│   │   └── predictions/
│   ├── raw/
│   │   ├── bird/
│   │   ├── cat/
│   │   └── dog/
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   └── source_splits.json
├── models/
│   └── audio_event_classifier.joblib
├── results/
│   ├── classifier_metrics.json
│   └── evaluation_results.json
├── src/
│   ├── prepare_dataset.py
│   ├── create_source_splits.py
│   ├── generate_split_contexts.py
│   ├── generate_qa.py
│   ├── train_audio_classifier.py
│   ├── audio_event_predictor.py
│   ├── context_builder.py
│   ├── question_classifier.py
│   ├── retriever.py
│   ├── answer_generator.py
│   ├── pipeline.py
│   └── evaluation.py
├── tests/
├── requirements.txt
├── README.md
└── report.md
```

## 6. Installation

Open PowerShell in the project root:

```powershell
cd "D:\Visual code\audio-context-layer"
```

Create/activate the virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Install Streamlit when required:

```powershell
pip install streamlit
```

## 7. Reproduce the Pipeline

### Create source-level splits

```powershell
python src/create_source_splits.py
```

### Generate source-disjoint contexts

```powershell
python src/generate_split_contexts.py
```

### Train the classifier

```powershell
python src/train_audio_classifier.py
```

### Generate event predictions

```powershell
Remove-Item .\data\processed\predictions\*.json -Force
python src/audio_event_predictor.py
```

### Build the context layer

```powershell
python src/context_builder.py
```

### Generate QA pairs

```powershell
python src/generate_qa.py
```

### Evaluate the end-to-end system

```powershell
python src/evaluation.py
```

The evaluation output is saved to:

```text
results/evaluation_results.json
```

## 8. Run the Streamlit Demo

From the project root:

```powershell
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

The demo contains three pages:

### Audio QA Demo
Select a generated context, play its WAV file, inspect the event timeline, and ask a natural-language question.

### Context Explorer
Inspect event counts, timestamps, durations, confidence values, sequence information, and the raw context JSON.

### Evaluation
Display the saved test-set metrics and the breakdown by question type.

## 9. Example Questions

```text
What sounds are present?
How many bird sound events occur?
How many cat sound events occur?
What happens after the first bird?
What happens before the second dog?
Which sound occurs first?
Which sound occurs last?
What is the sequence of sounds?
How long does the first bird sound last?
Why might a dog sound occur?
```

## 10. Current Evaluation

The current clean end-to-end test run produced:

| Metric | Result |
|---|---:|
| Question classification accuracy | 100.00% |
| Structured fact accuracy | 86.88% |
| Answer token F1 | 0.5249 |
| Pipeline errors | 0 |

Question-type structured fact accuracy:

| Type | Accuracy |
|---|---:|
| Causal | 100.00% |
| Duration | 93.33% |
| Counting | 83.33% |
| Temporal | 82.50% |
| Perceptual | 86.67% |
| Sequence | 73.33% |

The standalone source-disjoint audio classifier achieved 86.17% accuracy on the held-out source test set.

Literal answer exact match is not used as the primary quality indicator because generated answers may be semantically correct while using different wording from the templated reference answers.

## 11. Design Decisions

### Source-level splitting
Source files are split before multi-event contexts are generated so files used in test contexts do not appear in training contexts.

### Fixed one-second events
All context events are normalized to one second. This gives deterministic boundaries and makes temporal reasoning straightforward.

### Structured context layer
Instead of asking the answer generator to reason directly over raw audio, the system first creates structured event information. This makes counting, temporal reasoning, duration queries, and sequence questions explicit and inspectable.

### Lightweight classifier
MFCC mean/std features with a Random Forest provide a reproducible baseline that is practical for a small proof of concept.

### Occurrence-aware temporal questions
Repeated labels are represented as first/second/third occurrences to avoid ambiguous temporal questions.

### Conservative causal reasoning
The system does not claim a real-world cause from sound alone. Causal answers explicitly state that audio evidence alone cannot establish the real-world cause.

## 12. Limitations

- The multi-event audio contexts are synthetically concatenated rather than naturally occurring scenes.
- The current vocabulary contains only bird, cat, and dog.
- The dataset is small compared with large-scale audio resources.
- The classifier is a lightweight baseline rather than a large pretrained audio model.
- Some QA metrics are based on structured facts and template-aware heuristics rather than open-ended semantic judging.
- Causal evaluation checks whether the system preserves its causal limitation rather than measuring true causal inference.
- The current Streamlit demo operates on pre-generated contexts; arbitrary uploaded WAV processing can be added as a future enhancement.

## 13. Future Work

- Add direct WAV upload and automatic preprocessing.
- Expand the sound vocabulary.
- Replace the baseline classifier with a pretrained audio model.
- Add richer acoustic-scene reasoning.
- Add semantic answer evaluation.
- Add confidence-aware uncertainty handling.
- Evaluate on a larger and more natural audio QA dataset.

## 14. License / Dataset Note

This is an academic proof-of-concept project. Verify the license and attribution requirements for any source audio dataset before redistributing the raw recordings.
