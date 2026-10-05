# Predict the top-3 Fashion-MNIST classes for one image.
from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

def main():
    parser = argparse.ArgumentParser(description="Predict one image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="artifacts/prediction.png")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"]); model.eval().to(device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        probabilities = model(transform(image).unsqueeze(0).to(device)).softmax(1)[0]
    values, indices = probabilities.topk(3)
    print("Top-3 predictions:")
    for value, index in zip(values.tolist(), indices.tolist()): print(f"  {classes[index]}: {value:.1%}")
    plt.figure(figsize=(5, 5)); plt.imshow(image); plt.axis("off"); plt.title(f"{classes[indices[0]]}: {values[0]:.1%}"); plt.tight_layout(); plt.savefig(args.output, dpi=160)

if __name__ == "__main__":
    main()

