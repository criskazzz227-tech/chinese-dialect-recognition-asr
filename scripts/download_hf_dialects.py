import argparse
import csv
from pathlib import Path

from datasets import Audio, load_dataset


DATASETS = {
    "shanghai": "TingChen-ppmc/Shanghai_Dialect_Conversational_Speech_Corpus",
    "changsha": "TingChen-ppmc/Changsha_Dialect_Conversational_Speech_Corpus",
    "zhengzhou": "TingChen-ppmc/Zhengzhou_Dialect_Conversational_Speech_Corpus",
    "tianjin": "TingChen-ppmc/Tianjin_Dialect_Conversational_Speech_Corpus",
    "nanchang": "TingChen-ppmc/Nanchang_Dialect_Conversational_Speech_Corpus",
}


def safe_text(value):
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def download_dataset(label, dataset_id, limit, output_root):
    target_dir = output_root / label
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_id, split="train", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    rows = []
    for idx, item in enumerate(dataset):
        if idx >= limit:
            break

        audio = item["audio"]
        audio_bytes = audio.get("bytes")
        source_path = audio.get("path") or f"{idx:04d}.wav"
        suffix = Path(source_path).suffix.lower() or ".wav"
        filename = f"{idx + 1:04d}{suffix}"
        output_path = target_dir / filename

        if not audio_bytes:
            print(f"[WARN] skip {label} #{idx + 1}: no audio bytes")
            continue

        if not output_path.exists():
            output_path.write_bytes(audio_bytes)
            print(f"[OK] {label}: {filename}")
        else:
            print(f"[SKIP] {label}: {filename}")

        rows.append(
            {
                "path": output_path.as_posix(),
                "label": label,
                "dataset": dataset_id,
                "source_path": source_path,
                "speaker_id": safe_text(item.get("speaker_id")),
                "gender": safe_text(item.get("gender")),
                "transcription": safe_text(item.get("transcription")),
            }
        )

    return rows


def collect_existing_rows(output_root):
    rows = []
    for label_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        for audio_path in sorted(label_dir.glob("*.wav")):
            rows.append(
                {
                    "path": audio_path.as_posix(),
                    "label": label_dir.name,
                    "dataset": "",
                    "source_path": audio_path.name,
                    "speaker_id": "",
                    "gender": "",
                    "transcription": "",
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="samples per dialect")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "external" / "hf_dialects",
        help="output directory",
    )
    parser.add_argument(
        "--dialects",
        nargs="*",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="dialects to download",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for label in args.dialects:
        try:
            all_rows.extend(download_dataset(label, DATASETS[label], args.limit, args.output))
        except Exception as exc:
            print(f"[ERROR] {label}: {exc}")

    rows_by_path = {row["path"]: row for row in collect_existing_rows(args.output)}
    rows_by_path.update({row["path"]: row for row in all_rows})
    metadata_rows = sorted(rows_by_path.values(), key=lambda row: (row["label"], row["path"]))

    if not metadata_rows:
        print("No new metadata rows were collected; existing metadata was left unchanged.")
        return

    metadata_path = args.output / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label",
                "dataset",
                "source_path",
                "speaker_id",
                "gender",
                "transcription",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Saved {len(all_rows)} new metadata rows")
    print(f"Total metadata rows: {len(metadata_rows)}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
