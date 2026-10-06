# Experimental Results

## Experimental setup

- Dataset: Fashion-MNIST with ten clothing classes
- Model: ImageNet-pretrained ResNet-18
- Training duration: 1 epoch
- Batch size: 64
- Device: CPU
- Data split: 80% training, 10% validation, and 10% test

## Test results

| Metric | Result |
|---|---:|
| Test Accuracy | 92.3% |
| Test Macro-F1 | 92.2% |
| Validation Accuracy | 93.1% |
| Validation Macro-F1 | 93.0% |

## Interactive demonstration

Run:

```powershell
.\.venv\Scripts\python.exe src\demo.py --checkpoint artifacts\best.pt
```

After selecting an image, the program displays the original image and a whole-image prediction. `Analyze 4 regions` provides predictions for four image regions as a simple multi-region demonstration.

## Why a one-epoch training curve can look empty

The recorded result comes from a fast **1 epoch CPU run**. Consequently, `artifacts/history.csv` contains one training record and each curve has only one data point. A single point has no neighboring epoch to connect to, so an older plot may look empty or may not show a clear line. This does not indicate a failed training run and does not change the numeric values stored in `history.csv`.

The current `src/train.py` adds circular markers, axis labels, and a grid. To display a multi-point trend, run three epochs:

```powershell
.\.venv\Scripts\python.exe src\train.py --epochs 3 --batch-size 64 --output-dir artifacts
```

The resulting `artifacts/training_curves.png` will show changes in training loss, validation loss, and accuracy across epochs. The 92.3% test Accuracy and 92.2% test Macro-F1 reported in the application materials come from the saved test predictions; they are independent of whether the plot contains one point or several.

## Limitations and future work

The training data consists of 28×28 grayscale images containing one centered object. Performance on colorful product photographs, complex backgrounds, and multiple objects is therefore limited. The four-region feature demonstrates multi-region inference but is not a formal object detector.

Future work could use a color clothing dataset and an object detection model such as YOLO or SSD to output a bounding box, class, and confidence for each object.
