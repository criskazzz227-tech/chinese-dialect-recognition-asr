import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .config import (
    DIALECTS,
    DISPLAY_NAMES,
    EXPERIMENT_RESULTS_PATH,
    OUT_DIR,
    TEST_CSV,
)
from .logging_config import configure_logging
from .pipeline import CLASSIFIER_OPTIONS, run_pipeline


def evaluate_classifier(classifier):
    frame = pd.read_csv(TEST_CSV)
    true_labels = []
    predicted_labels = []
    rows = []

    for record in frame.itertuples(index=False):
        output = run_pipeline(
            record.path,
            classifier=classifier,
            run_asr=False,
        )
        true_labels.append(record.label)
        predicted_labels.append(output["dialect"])
        rows.append(
            {
                "path": record.path,
                "ground_truth": record.label,
                "prediction": output["dialect"],
                "confidence": output["confidence"],
                "classifier": classifier,
            }
        )

    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro")
        ),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=DIALECTS,
            output_dict=True,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=DIALECTS,
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=DIALECTS,
    )
    display.plot(cmap="Blues", colorbar=False)
    plt.title(f"{classifier.upper()} Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / f"{classifier}_confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"{classifier}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return metrics


def evaluate_all(classifiers):
    results = {
        "dataset": {
            "train_samples": len(pd.read_csv("data/train.csv")),
            "test_samples": len(pd.read_csv(TEST_CSV)),
            "classes": DIALECTS,
            "split": "stratified random split, seed=42",
            "limitation": (
                "Speaker-disjoint evaluation is unavailable because most "
                "downloaded metadata lacks speaker identifiers."
            ),
        },
        "models": {},
    }
    for classifier in classifiers:
        print(f"Evaluating {CLASSIFIER_OPTIONS[classifier]}...")
        results["models"][classifier] = evaluate_classifier(classifier)
        print(
            f"{classifier}: "
            f"accuracy={results['models'][classifier]['accuracy']:.4f}, "
            f"macro_f1={results['models'][classifier]['macro_f1']:.4f}"
        )

    EXPERIMENT_RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=list(CLASSIFIER_OPTIONS),
        default=list(CLASSIFIER_OPTIONS),
    )
    args = parser.parse_args()
    evaluate_all(args.classifiers)
