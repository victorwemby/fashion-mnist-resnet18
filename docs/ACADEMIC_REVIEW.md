# Academic review guide

## Research question

How effectively can ImageNet transfer learning adapt ResNet-18 to Fashion-MNIST, and what changes when data augmentation or backbone fine-tuning is removed?

## Evidence included

- `artifacts/training_curves.png`: optimization behavior.
- `artifacts/confusion_matrix.png`: class-level errors.
- `artifacts/ablation/ablation_results.csv`: controlled comparisons.
- `reports/results.md`: recorded metrics and limitations.
- `src/gradcam.py`: visual explanation of model attention.
- `src/demo.py`: interactive inference interface.

## Reproduction

Install dependencies and run the commands in `README.md`. Torchvision downloads the four Fashion-MNIST files automatically into `data/FashionMNIST/raw/`; manual placement of those files is optional when network access is unavailable.

