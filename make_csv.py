import csv
import random
from pathlib import Path

ROOT = Path("data")
EXTERNAL_ROOT = ROOT / "external" / "hf_dialects"
OUT_TRAIN = ROOT / "train.csv"
OUT_TEST = ROOT / "test.csv"

DIALECTS = ["shanghai", "changsha", "zhengzhou", "tianjin", "nanchang"]
TEST_RATIO = 0.2
RANDOM_SEED = 42
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def collect_rows():
    rows = []
    for dialect in DIALECTS:
        dialect_dir = EXTERNAL_ROOT / dialect
        if not dialect_dir.is_dir():
            print(f"[WARN] Skip {dialect}, folder not found: {dialect_dir}")
            continue

        files = sorted(
            path
            for path in dialect_dir.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
        for path in files:
            rows.append([path.as_posix(), dialect])
        print(f"[INFO] {dialect}: {len(files)} files")
    return rows


def stratified_split(rows):
    by_label = {}
    for path, label in rows:
        by_label.setdefault(label, []).append([path, label])

    rng = random.Random(RANDOM_SEED)
    train_rows, test_rows = [], []
    for label, label_rows in sorted(by_label.items()):
        rng.shuffle(label_rows)
        n_test = max(1, int(len(label_rows) * TEST_RATIO)) if len(label_rows) > 1 else 0
        test_rows.extend(label_rows[:n_test])
        train_rows.extend(label_rows[n_test:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)


def main():
    rows = collect_rows()
    if not rows:
        raise RuntimeError(f"No audio files found under {EXTERNAL_ROOT}")

    train_rows, test_rows = stratified_split(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_TRAIN, train_rows)
    write_csv(OUT_TEST, test_rows)

    print(f"Generated {len(train_rows)} train samples")
    print(f"Generated {len(test_rows)} test samples")


if __name__ == "__main__":
    main()
