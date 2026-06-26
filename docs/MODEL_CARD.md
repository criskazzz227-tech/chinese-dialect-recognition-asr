# Model Card

## Intended Use

This project compares two classifiers for five Chinese regional speech categories: Shanghai, Changsha, Zhengzhou, Tianjin, and Nanchang. It is intended for coursework, experimentation, and portfolio demonstration.

## Models

- Baseline: 80-bin Log-Mel features with CMVN and a 1D CNN.
- Transfer features: frozen Whisper base encoder, temporal mean/std pooling, StandardScaler, and Logistic Regression.

## Evaluation

The current dataset contains 250 clips and uses a stratified random 80/20 split with seed 42.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Log-Mel CNN | 0.680 | 0.679 |
| Whisper Encoder + LR | 0.980 | 0.980 |

## Limitations

- The dataset is very small.
- Most source metadata lacks speaker identifiers.
- Changsha clips in the downloaded subset belong to one recorded speaker.
- Random splitting may leak speaker, channel, or corpus characteristics.
- The confidence scores are not formally calibrated.
- The model must not be described as a universal Chinese dialect recognizer.

## Recommended Next Evaluation

Collect multiple speakers per class, preserve speaker IDs, use speaker-disjoint and cross-device splits, report calibration error, and add out-of-distribution audio for rejection testing.
