import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
import joblib


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "source_splits.json"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "audio_event_classifier.joblib"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "classifier_metrics.json"
)

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# AUDIO SETTINGS
# ==========================================================

SAMPLE_RATE = 16000

N_MFCC = 40


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(audio_path):
    """
    Extract MFCC mean and standard deviation features.
    """

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mean_features = np.mean(
        mfcc,
        axis=1
    )

    std_features = np.std(
        mfcc,
        axis=1
    )

    features = np.concatenate(
        [
            mean_features,
            std_features
        ]
    )

    return features


# ==========================================================
# LOAD SOURCE SPLITS
# ==========================================================

def load_source_splits():

    with open(
        SPLIT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["splits"]


# ==========================================================
# BUILD DATASET FOR ONE SPLIT
# ==========================================================

def load_split_data(
    splits,
    split_name
):

    X = []
    y = []

    for class_name in sorted(
        splits[split_name]
    ):

        files = splits[
            split_name
        ][class_name]

        for relative_path in files:

            audio_path = (
                PROJECT_ROOT
                / relative_path
            )

            if not audio_path.exists():

                print(
                    f"WARNING: Missing file: "
                    f"{audio_path}"
                )

                continue

            try:

                features = extract_features(
                    audio_path
                )

                X.append(features)
                y.append(class_name)

            except Exception as error:

                print(
                    f"WARNING: Could not process "
                    f"{audio_path}"
                )

                print(
                    f"Reason: {error}"
                )

    return np.array(X), np.array(y)


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate_model(
    model,
    X,
    y,
    split_name
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0
    )

    print()
    print(
        f"{split_name.upper()} RESULTS"
    )

    print(
        f"Samples: {len(y)}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy),
        "classification_report": report
    }


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 65)
    print("AUDIO EVENT CLASSIFIER")
    print("SOURCE-DISJOINT TRAINING")
    print("=" * 65)

    # ------------------------------------------------------
    # Load source manifest
    # ------------------------------------------------------

    splits = load_source_splits()

    # ------------------------------------------------------
    # Build TRAIN dataset
    # ------------------------------------------------------

    print()
    print("Loading TRAIN sources...")

    X_train, y_train = load_split_data(
        splits,
        "train"
    )

    # ------------------------------------------------------
    # Build VALIDATION dataset
    # ------------------------------------------------------

    print(
        "Loading VALIDATION sources..."
    )

    X_validation, y_validation = (
        load_split_data(
            splits,
            "validation"
        )
    )

    # ------------------------------------------------------
    # Build TEST dataset
    # ------------------------------------------------------

    print(
        "Loading TEST sources..."
    )

    X_test, y_test = load_split_data(
        splits,
        "test"
    )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print()
    print(
        f"Training samples:   {len(X_train)}"
    )

    print(
        f"Validation samples: {len(X_validation)}"
    )

    print(
        f"Test samples:       {len(X_test)}"
    )

    print(
        f"Feature size:       {X_train.shape[1]}"
    )

    # ------------------------------------------------------
    # Train ONLY on training sources
    # ------------------------------------------------------

    print()
    print(
        "Training Random Forest "
        "using TRAIN sources only..."
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model training completed."
    )

    # ------------------------------------------------------
    # Validation evaluation
    # ------------------------------------------------------

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "validation"
    )

    # ------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "test"
    )

    # ------------------------------------------------------
    # Save model
    # ------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    # ------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------

    metrics = {

        "experiment": (
            "source_disjoint_audio_classifier"
        ),

        "feature_type": (
            "MFCC mean + standard deviation"
        ),

        "sample_rate": SAMPLE_RATE,

        "n_mfcc": N_MFCC,

        "n_estimators": 200,

        "random_seed": 42,

        "training_samples": int(
            len(X_train)
        ),

        "validation_samples": int(
            len(X_validation)
        ),

        "test_samples": int(
            len(X_test)
        ),

        "validation": validation_metrics,

        "test": test_metrics
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    # ------------------------------------------------------
    # Complete
    # ------------------------------------------------------

    print()
    print("=" * 65)
    print("SOURCE-DISJOINT CLASSIFIER TRAINING COMPLETED")
    print("=" * 65)

    print(
        "Model saved to:"
    )

    print(
        MODEL_FILE
    )

    print()

    print(
        "Metrics saved to:"
    )

    print(
        METRICS_FILE
    )


if __name__ == "__main__":
    main()