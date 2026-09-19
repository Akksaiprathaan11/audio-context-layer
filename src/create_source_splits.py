import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_FILE = PROJECT_ROOT / "data" / "source_splits.json"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def main():

    print("=" * 60)
    print("SOURCE-LEVEL DATASET SPLIT")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    splits = {
        "train": {},
        "validation": {},
        "test": {}
    }

    total_files = 0

    for class_dir in sorted(RAW_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        wav_files = sorted(
            class_dir.glob("*.wav")
        )

        if not wav_files:
            continue

        files = [
            str(
                path.relative_to(PROJECT_ROOT)
            )
            for path in wav_files
        ]

        random.shuffle(files)

        total = len(files)

        train_end = int(
            total * TRAIN_RATIO
        )

        validation_end = (
            train_end
            + int(total * VALIDATION_RATIO)
        )

        train_files = files[:train_end]

        validation_files = files[
            train_end:validation_end
        ]

        test_files = files[
            validation_end:
        ]

        splits["train"][
            class_dir.name
        ] = train_files

        splits["validation"][
            class_dir.name
        ] = validation_files

        splits["test"][
            class_dir.name
        ] = test_files

        total_files += total

        print()
        print(
            f"Class: {class_dir.name}"
        )

        print(
            f"  Total:      {total}"
        )

        print(
            f"  Train:      {len(train_files)}"
        )

        print(
            f"  Validation: {len(validation_files)}"
        )

        print(
            f"  Test:       {len(test_files)}"
        )

    output = {
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO
        },
        "total_files": total_files,
        "splits": splits
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print()
    print("=" * 60)
    print(
        f"Total source files: {total_files}"
    )

    print(
        f"Split manifest saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 60)


if __name__ == "__main__":
    main()