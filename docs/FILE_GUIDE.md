# Project file guide

This page gives a complete, beginner-friendly description of the files that should be visible to a reviewer on GitHub.

Items marked **(Admissions priority)** are the files a reviewer should open first.

## Root files

| File | Role |
|---|---|
| `README.md` | Project overview, results, setup commands, and reading order |
| `requirements.txt` | Python packages used by the project |
| `.gitignore` | Excludes local environments, downloaded data, and model checkpoints; the demo video is tracked with Git LFS |
| `LICENSE` | MIT license for the code |
| `CONTRIBUTING.md` | Rules for making future changes |
| `SECURITY.md` | Security reporting guidance |

## Source code

| File | Role |
|---|---|
| `src/data.py` | Downloads Fashion-MNIST and builds training, validation, and test loaders |
| `src/train.py` | Runs the main training and evaluation experiment (Admissions priority) |
| `src/infer.py` | Predicts one user-provided image from PowerShell |
| `src/demo.py` | Opens the interactive desktop interface |
| `src/gradcam.py` | Produces a heatmap of image regions used by the model (Admissions priority) |
| `src/ablation.py` | Compares full fine-tuning, no augmentation, and frozen-backbone settings (Admissions priority) |
| `src/utils.py` | Shared reproducibility and directory helpers |
| `src/verify_results.py` | Recomputes the reported metrics from saved predictions and writes an evidence CSV (Admissions priority) |

## Evidence and reports

| Location | Role |
|---|---|
| `artifacts/` | Plots and CSV files produced by the recorded experiment (Admissions priority) |
| `artifacts/ablation/` | Per-setting ablation outputs and summary table |
| `reports/results.md` | Numeric results and limitations (Admissions priority) |
| `reports/interpretability_ablation.md` | Explanation of Grad-CAM and ablation in plain English (Admissions priority) |
| `reports/gradcam_case_d02d510d.md` | Case study for the example image |
| `reports/learning_guide.md` | Beginner learning notes |
| `reports/results_template.md` | Template for a future run |
| `reports/evidence_checklist.md` | Claims, source files, and reproducible verification commands (Admissions priority) |

## Supporting material

| Location | Role |
|---|---|
| `docs/` | Reviewer-facing documentation, including this guide |
| `examples/` | Public example images and usage notes |
| `data/README.md` | Explains the runtime dataset location; raw data is not uploaded |
| `.github/workflows/smoke-test.yml` | Automated syntax/import check on GitHub |

## Generated evidence files

| File | Role |
|---|---|
| `artifacts/metrics_evidence.csv` | Four headline metrics and the source file used for each value (Admissions priority) |
| `artifacts/metrics_evidence_verification_screenshot.png` | Screenshot showing the metric verification command and output (Admissions priority) |
| `artifacts/history.csv` | Epoch-level training and validation metrics |
| `artifacts/ablation/ablation_results.csv` | Validation comparison for the three ablation settings (Admissions priority) |
| `artifacts/confusion_matrix.png` | Visual summary of class-level errors (Admissions priority) |
| `artifacts/gradcam_d02d510d1a241b34bb4ef477ea5bdf9b.png` | Grad-CAM visual explanation for the documented example (Admissions priority) |

## Deliberately excluded from GitHub

`.venv/`, `data/FashionMNIST/`, and `artifacts/*.pt`/`artifacts/**/*.pt` are local or large files excluded by `.gitignore`. `demonstration.mp4` is large but is tracked with Git LFS and is marked as the practical demonstration above.


