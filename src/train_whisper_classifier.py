import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config import APP, DIALECTS, OUT_DIR, TEST_CSV, TRAIN_CSV
from .logging_config import configure_logging
from .whisper_classifier import (
    extract_whisper_embedding,
    load_whisper_encoder,
    save_classifier,
)


logger = logging.getLogger(__name__)


def extract_split(csv_path, split_name, feature_model, rebuild_cache=False):
    cache_path = OUT_DIR / f"whisper_{feature_model}_{split_name}_features.joblib"
    if cache_path.exists() and not rebuild_cache:
        logger.info("Loading cached features from %s", cache_path)
        return joblib.load(cache_path)

    frame = pd.read_csv(csv_path)
    model, device = load_whisper_encoder(feature_model)
    features = []
    for path in tqdm(frame["path"], desc=f"Whisper features: {split_name}"):
        features.append(
            extract_whisper_embedding(path, model=model, device=device)
        )

    payload = {
        "features": np.stack(features),
        "labels": frame["label"].to_numpy(),
        "paths": frame["path"].to_numpy(),
    }
    joblib.dump(payload, cache_path)
    logger.info("Saved feature cache to %s", cache_path)
    return payload


def train(feature_model=None, rebuild_cache=False):
    feature_model = feature_model or APP.whisper_feature_model
    train_data = extract_split(
        TRAIN_CSV,
        "train",
        feature_model,
        rebuild_cache,
    )
    test_data = extract_split(
        TEST_CSV,
        "test",
        feature_model,
        rebuild_cache,
    )

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    classifier.fit(train_data["features"], train_data["labels"])
    predictions = classifier.predict(test_data["features"])
    metrics = {
        "accuracy": float(
            accuracy_score(test_data["labels"], predictions)
        ),
        "macro_f1": float(
            f1_score(test_data["labels"], predictions, average="macro")
        ),
        "classification_report": classification_report(
            test_data["labels"],
            predictions,
            labels=DIALECTS,
            output_dict=True,
            zero_division=0,
        ),
    }
    save_classifier(classifier, metrics, feature_model)

    report_path = OUT_DIR / "whisper_classifier_metrics.json"
    report_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Train a dialect classifier on frozen Whisper features."
    )
    parser.add_argument(
        "--feature-model",
        default=APP.whisper_feature_model,
        choices=["tiny", "base", "small"],
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    train(args.feature_model, args.rebuild_cache)
