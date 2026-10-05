# Launch the interactive desktop inference demo.
from __future__ import annotations
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageOps, ImageTk
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

class FashionDemo:
    def __init__(self, root: tk.Tk, checkpoint: str):
        self.root = root; self.root.title("Fashion-MNIST ResNet-18 Demo"); self.root.geometry("780x760")
        device = torch.device("cpu"); data = torch.load(checkpoint, map_location=device)
        self.model = resnet18(weights=None); self.model.fc = nn.Linear(self.model.fc.in_features, 10); self.model.load_state_dict(data["model"]); self.model.eval()
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.image_canvas = tk.Canvas(root, width=560, height=380, bg="#eeeeee", highlightthickness=0)
        self.image_canvas.pack(pady=16)
        self.image_canvas.create_text(280, 190, text="Choose an image to begin", fill="#222222", font=("Segoe UI", 13))
        tk.Label(root, text="Color is preserved for display. Predictions use grayscale crops, matching Fashion-MNIST training.", fg="#555555").pack()
        tk.Button(root, text="Choose image", command=self.choose_image, width=20).pack(pady=8)
        tk.Button(root, text="Analyze 4 regions", command=self.analyze_regions, width=20).pack(pady=2)
        self.result = tk.Label(root, text="", justify="left", font=("Segoe UI",13)); self.result.pack(pady=12)
    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if not path: return
        try:
            image = Image.open(path).convert("RGB")
            self.current_image = image
            preview = ImageOps.contain(image, (560, 380), method=Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (560, 380), "white")
            canvas.paste(preview, ((560 - preview.width) // 2, (380 - preview.height) // 2))
            self.preview = ImageTk.PhotoImage(canvas)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(280, 190, image=self.preview, anchor="center")
            model_image = ImageOps.fit(image, (224, 224), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)).convert("L")
            model_image = ImageEnhance.Contrast(model_image).enhance(1.5).convert("RGB")
            with torch.no_grad(): probability = self.model(self.transform(model_image).unsqueeze(0)).softmax(1)[0]
            values, indices = probability.topk(3)
            lines = [f"Prediction: {CLASSES[indices[0]]} ({values[0].item():.1%})", "", "Top-3:"]
            lines += [f"{i}. {CLASSES[index]} — {value.item():.1%}" for i, (value, index) in enumerate(zip(values, indices), 1)]
            self.result.configure(text="\n".join(lines), fg="#17365d")
        except Exception as error: messagebox.showerror("Prediction failed",str(error))

    def analyze_regions(self):
        if not hasattr(self, "current_image"):
            messagebox.showinfo("Choose an image first", "Click Choose image before analyzing regions.")
            return
        try:
            image = self.current_image
            # Analyze the full image and four regions. This is a simple multi-region mode; it is not a trained detector.
            w, h = image.size
            regions = [("whole image", image), ("top-left", image.crop((0, 0, w//2, h//2))), ("top-right", image.crop((w//2, 0, w, h//2))), ("bottom-left", image.crop((0, h//2, w//2, h))), ("bottom-right", image.crop((w//2, h//2, w, h)))]
            lines = ["Multi-region predictions:"]
            batch = []
            for _, region in regions:
                model_image = ImageOps.fit(region, (224, 224), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)).convert("L")
                model_image = ImageEnhance.Contrast(model_image).enhance(1.5).convert("RGB")
                batch.append(self.transform(model_image))
            with torch.no_grad(): probs = self.model(torch.stack(batch)).softmax(1)
            for (name, _), probability in zip(regions, probs):
                value, index = probability.max(0)
                lines.append(f"{name}: {CLASSES[index]} ({value.item():.1%})")
            lines.append("\nNote: multiple objects require an object-detection model for reliable boxes.")
            self.result.configure(text="\n".join(lines), fg="#17365d")
        except Exception as error: messagebox.showerror("Region analysis failed",str(error))
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",default="artifacts/best.pt"); args=parser.parse_args()
    if not Path(args.checkpoint).exists(): raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    root=tk.Tk(); FashionDemo(root,args.checkpoint); root.mainloop()
if __name__=="__main__": main()

