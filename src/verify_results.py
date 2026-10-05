"""Recompute headline metrics from the saved prediction CSV.

This script does not download data or retrain the model. It provides an
auditable check of the reported test accuracy and macro-F1.
"""

import argparse
from pathlib import Path

import pandas as pd


def macro_f1_score(true_labels: pd.Series, predicted_labels: pd.Series) -> float:
    classes = sorted(set(true_labels) | set(predicted_labels))
    scores = []
    for label in classes:
        tp = sum((true_labels == label) & (predicted_labels == label))
        fp = sum((true_labels != label) & (predicted_labels == label))
        fn = sum((true_labels == label) & (predicted_labels != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="artifacts/test_predictions.csv")
    parser.add_argument("--output", default="artifacts/test_metrics.csv")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    accuracy = (predictions["true"] == predictions["pred"]).mean()
    macro_f1 = macro_f1_score(predictions["true"], predictions["pred"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"metric": "test_accuracy", "value": accuracy},
         {"metric": "test_macro_f1", "value": macro_f1}]
    ).to_csv(output, index=False)

    print(f"Predictions: {len(predictions)}")
    print(f"Test Accuracy: {accuracy * 100:.1f}%")
    print(f"Test Macro-F1: {macro_f1 * 100:.1f}%")
    print(f"Saved evidence: {output}")


if __name__ == "__main__":
    main()
