# Interpretability and Ablation Study

## Grad-CAM

Grad-CAM uses gradients from the final convolutional layer to create a heatmap showing which image regions contributed to the classification decision.

## Ablation study

The study compares training configurations using the same data split and training duration. Only one design choice is changed at a time where possible.

The three settings are:

- `full_finetuning`: full fine-tuning with data augmentation
- `no_augmentation`: full fine-tuning without random flips and crops
- `frozen_backbone`: the ResNet-18 backbone is frozen and only the final classifier is trained

Results are saved to `artifacts/ablation/ablation_results.csv`.

The summary CSV records validation Accuracy and Macro-F1. The corresponding
`test_predictions.csv` and `classification_report.csv` files inside each setting
directory provide the test-set evidence.

Interpretation should focus on differences in validation Accuracy and Macro-F1. A formal study should use multiple random seeds or more epochs for more stable estimates.
