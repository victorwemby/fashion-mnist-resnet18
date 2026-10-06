# Grad-CAM Case Study: d02d510d1a241b34bb4ef477ea5bdf9b

## Input

The original input was selected from the Desktop. A portable copy is included in the repository at `examples/d02d510d1a241b34bb4ef477ea5bdf9b.png`.

## Generation command

```powershell
.\.venv\Scripts\python.exe src\gradcam.py `
  --checkpoint artifacts\best.pt `
  --image examples\d02d510d1a241b34bb4ef477ea5bdf9b.png `
  --output artifacts\gradcam_d02d510d1a241b34bb4ef477ea5bdf9b.png
```

## Interpretation

Grad-CAM projects gradients from the final ResNet-18 convolutional layer back into image space. Red regions contributed more strongly to the prediction, while blue regions contributed less. Check whether attention is concentrated on the clothing item rather than the background.

## Application wording

`I used Grad-CAM to inspect whether the classifier focused on the clothing object when making its prediction. The visualization provided a qualitative check of the model's decision process and exposed potential sensitivity to background and input distribution.`

## Limitation

Fashion-MNIST consists of 28×28 grayscale images. A heatmap for a colorful product photograph is therefore a distribution-shift example and does not demonstrate general-purpose product recognition.

