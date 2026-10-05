# Evidence checklist for application materials

Use the following files as evidence. The values in the application description
must match these files and the command output.

| Claim | Evidence | How to verify |
|---|---|---|
| Validation Accuracy 93.1% | `artifacts/history.csv` | Read the `val_accuracy` column: `0.931` |
| Validation Macro-F1 93.0% | `artifacts/history.csv` | Read the `val_macro_f1` column: `0.9300036` |
| Test Accuracy 92.3% | `artifacts/test_predictions.csv` | Run `src/verify_results.py` |
| Test Macro-F1 92.2% | `artifacts/test_predictions.csv` | Run `src/verify_results.py` |
| Ablation comparison | `artifacts/ablation/ablation_results.csv` | Read the three recorded settings |
| Visual model evidence | `artifacts/confusion_matrix.png`, `artifacts/gradcam_*.png` | Open the PNG files |

## Screenshot procedure

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe src\verify_results.py
Get-Content artifacts\test_metrics.csv
Get-Content artifacts\history.csv
Get-Content artifacts\ablation\ablation_results.csv
```

Take one screenshot showing the command output and one screenshot showing the
corresponding CSV files in GitHub. This provides both a reproducible calculation
and a public repository record.
