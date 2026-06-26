import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ASSETS = ROOT / "docs" / "assets"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    results = json.loads(
        (OUTPUTS / "experiment_results.json").read_text(encoding="utf-8")
    )

    model_keys = ["cnn", "whisper"]
    labels = ["Log-Mel CNN", "Whisper Encoder + LR"]
    accuracy = [
        results["models"][key]["accuracy"] * 100 for key in model_keys
    ]
    macro_f1 = [
        results["models"][key]["macro_f1"] * 100 for key in model_keys
    ]

    positions = range(len(model_keys))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.bar(
        [position - width / 2 for position in positions],
        accuracy,
        width,
        label="Accuracy",
        color="#0f766e",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        macro_f1,
        width,
        label="Macro-F1",
        color="#d97706",
    )
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Score (%)")
    axis.set_title("Dialect Classifier Comparison")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f", padding=3)
    figure.tight_layout()
    figure.savefig(ASSETS / "model_comparison.png", dpi=180)
    plt.close(figure)

    for name in ("cnn_confusion_matrix.png", "whisper_confusion_matrix.png"):
        shutil.copy2(OUTPUTS / name, ASSETS / name)


if __name__ == "__main__":
    main()
