# Examples

These are small, public demo inputs used by the command-line and desktop demos.

| File | Use |
|---|---|
| `d02d510d1a241b34bb4ef477ea5bdf9b.png` | Main Grad-CAM case study image |
| `685f7e55e671e631c2e4eda0c181fd1a.png` | Additional single-image prediction example |
| `download.png` | Extra visual example for manual testing |

To test an image from PowerShell:

```powershell
.\.venv\Scripts\python.exe src\infer.py --checkpoint artifacts\best.pt --image examples\d02d510d1a241b34bb4ef477ea5bdf9b.png
```

Do not add private photos or large binary files to this folder.
