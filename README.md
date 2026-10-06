# Fashion-MNIST ResNet-18: Interpretable Image Classification

A reproducible computer vision project for academic review. It adapts an ImageNet-pretrained ResNet-18 to ten-class Fashion-MNIST classification and includes evaluation, ablation experiments, Grad-CAM interpretation, and an interactive desktop demo.

## Research question

How does transfer learning adapt a CNN to Fashion-MNIST, and how much do augmentation and backbone fine-tuning affect performance?

## Results

The recorded one-epoch CPU run achieved **92.3% test accuracy** and **92.2% macro-F1**. In the ablation runs, freezing the backbone reduced the frozen run's test accuracy to **77.0%**, showing the value of task-specific fine-tuning. The summary CSV records validation metrics; each ablation folder also contains its test predictions and reports.

## Repository guide

The files are grouped by role so that a reviewer can read the project in this order:

### 1. Source code (`src/`)

| File | What it does |
|---|---|
| `src/train.py` | Main experiment: train, validate, test, and save metrics and plots |
| `src/data.py` | Download Fashion-MNIST, apply transforms, and create fixed data splits |
| `src/infer.py` | Predict the top-3 classes for one image from the command line |
| `src/demo.py` | Launch the Tkinter desktop demo for image selection and prediction |
| `src/gradcam.py` | Create a Grad-CAM heatmap showing which image regions affect a prediction |
| `src/ablation.py` | Run controlled comparisons: full fine-tuning, no augmentation, and frozen backbone |
| `src/utils.py` | Shared random-seed and directory utilities |
| `src/verify_results.py` | Recompute test metrics from saved predictions for auditability |

### 2. Results and reports

| Path | What it contains |
|---|---|
| `artifacts/` | Generated plots, CSV metrics, predictions, and the Grad-CAM example |
| `artifacts/ablation/` | Outputs from each ablation configuration and the summary CSV |
| `reports/results.md` | Measured accuracy, Macro-F1, and experiment interpretation |
| `reports/interpretability_ablation.md` | Plain-English explanation of Grad-CAM and the ablation study |
| `reports/gradcam_case_d02d510d.md` | Short case study for the supplied example image |
| `reports/learning_guide.md` | Beginner-friendly explanation of the complete workflow |
| `reports/results_template.md` | Blank template for recording a future run |
| `reports/evidence_checklist.md` | Exact evidence files and commands for verifying reported metrics |

### 3. Documentation and examples

| Path | What it contains |
|---|---|
| `docs/REPOSITORY_MAP.md` | One-page map of the repository for reviewers |
| `docs/ACADEMIC_REVIEW.md` | Project motivation, reproducibility notes, and limitations |
| `docs/FILE_GUIDE.md` | Complete file-by-file guide for GitHub reviewers |
| `examples/` | Public demo images and their usage notes; no private photos are required |
| `data/README.md` | Explains where Fashion-MNIST is downloaded; the dataset itself is ignored by Git |

### 4. Project configuration

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies needed to reproduce the experiments |
| `.gitignore` | Prevents `.venv/`, downloaded data, and model checkpoints from being uploaded |
| `.github/workflows/smoke-test.yml` | GitHub Actions check that imports and compiles the source code |
| `LICENSE` | MIT open-source license |
| `CONTRIBUTING.md` | Small guide for future improvements |
| `SECURITY.md` | Contact and reporting guidance for security issues |
| `demonstration.mp4` | Screen recording of the desktop demo, stored with Git LFS |

The local `.venv/`, `data/`, and `*.pt`/`*.pth` checkpoint files are intentionally kept out of GitHub. The demonstration video is tracked with Git LFS because it is a large binary file. A reviewer can recreate the software demo by following the commands below.

## Quick start (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\train.py --epochs 1 --batch-size 64 --output-dir artifacts
```

The first run downloads Fashion-MNIST and the ResNet-18 weights. Use `--weights path\to\resnet18-f37072fd.pth` to load a local checkpoint.

## Analysis commands

```powershell
.\.venv\Scripts\python.exe src\gradcam.py --checkpoint artifacts\best.pt --image path\to\image.png --output artifacts\gradcam.png
.\.venv\Scripts\python.exe src\ablation.py --runs 1 --output-dir artifacts\ablation
.\.venv\Scripts\python.exe src\demo.py --checkpoint artifacts\best.pt
.\.venv\Scripts\python.exe src\verify_results.py
```

The verification command recomputes the two test metrics from the saved
predictions and reads the two validation metrics from `artifacts/history.csv`.
It writes all four values and their source paths to `artifacts/metrics_evidence.csv`.

## Recommended reading order

1. Read this README for the research question and headline results.
2. Open `reports/results.md` to inspect the measured metrics.
3. Open the plots in `artifacts/` and the Grad-CAM case in `reports/gradcam_case_d02d510d.md`.
4. Read `src/train.py` and `src/ablation.py` to see how the experiments were implemented.
5. Run the desktop demo with `src/demo.py` for an interactive presentation.

## Limitations

Fashion-MNIST contains centered 28×28 grayscale single-object images. Performance on complex color photographs or multiple objects is therefore a distribution-shift case. Reliable multi-object recognition would require a color detection dataset and an object detector.

## License

MIT. See `LICENSE`.
