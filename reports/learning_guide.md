# Beginner Learning Guide

## Step 1: Understand the task

The model receives one image and selects one of ten Fashion-MNIST classes: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, or ankle boot.

## Step 2: Run training successfully

First use one epoch to verify the environment:

```powershell
.\.venv\Scripts\python.exe src\train.py --epochs 1 --batch-size 64
```

After the pipeline works, a longer run can be used for a smoother learning curve.

## Step 3: Learn four core concepts

- **Training set:** data used by the model to learn parameters.
- **Validation set:** data used after each epoch to monitor progress.
- **Test set:** data used for the final evaluation and kept separate from tuning.
- **Epoch:** one complete pass through the training set.

## Step 4: Inspect the results

Open `artifacts/history.csv` for numeric values. Open the PNG files for the training curves and confusion matrix. Check whether validation Accuracy and Macro-F1 improve and which classes are confused most often.

## Step 5: Run an inference demo

Prepare an image and run the inference command in the README. The model was trained on centered, low-resolution grayscale images, so an internet photograph may receive low confidence. That is expected distribution shift and can be discussed as part of error analysis.

## Step 6: Write the application description

Report only measured results. Record the training time, device (CPU or GPU), test Accuracy, Macro-F1, the most confused class pair, and one prediction visualisation.
