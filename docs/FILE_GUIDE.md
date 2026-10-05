# Project file guide

This page gives a complete, beginner-friendly description of the files that should be visible to a reviewer on GitHub.

## Root files

| File | Role |
|---|---|
| `README.md` | Project overview, results, setup commands, and reading order |
| `requirements.txt` | Python packages used by the project |
| `.gitignore` | Excludes local environments, downloaded data, checkpoints, and the large demo video |
| `LICENSE` | MIT license for the code |
| `CONTRIBUTING.md` | Rules for making future changes |
| `SECURITY.md` | Security reporting guidance |

## Source code

| File | Role |
|---|---|
| `src/data.py` | Downloads Fashion-MNIST and builds training, validation, and test loaders |
| `src/train.py` | Runs the main training and evaluation experiment |
| `src/infer.py` | Predicts one user-provided image from PowerShell |
| `src/demo.py` | Opens the interactive desktop interface |
| `src/gradcam.py` | Produces a heatmap of image regions used by the model |
| `src/ablation.py` | Compares full fine-tuning, no augmentation, and frozen-backbone settings |
| `src/utils.py` | Shared reproducibility and directory helpers |

## Evidence and reports

| Location | Role |
|---|---|
| `artifacts/` | Plots and CSV files produced by the recorded experiment |
| `artifacts/ablation/` | Per-setting ablation outputs and summary table |
| `reports/results.md` | Numeric results and limitations |
| `reports/interpretability_ablation.md` | Explanation of Grad-CAM and ablation in plain English |
| `reports/gradcam_case_d02d510d.md` | Case study for the example image |
| `reports/learning_guide.md` | Beginner learning notes |
| `reports/results_template.md` | Template for a future run |

## Supporting material

| Location | Role |
|---|---|
| `docs/` | Reviewer-facing documentation, including this guide |
| `examples/` | Public example images and usage notes |
| `data/README.md` | Explains the runtime dataset location; raw data is not uploaded |
| `.github/workflows/smoke-test.yml` | Automated syntax/import check on GitHub |

## Deliberately excluded from GitHub

`.venv/`, `data/FashionMNIST/`, `artifacts/*.pt`, `artifacts/**/*.pt`, and `demonstration.mp4` are local or large files. They are reproducible or viewable through the source code and are excluded by `.gitignore`.
