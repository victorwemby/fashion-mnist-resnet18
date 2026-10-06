# Train and evaluate the Fashion-MNIST ResNet-18 classifier.
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm
from data import CLASSES, make_datasets
from utils import ensure_dir, seed_everything

def accuracy_score(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))

def macro_f1(y_true, y_pred, num_classes):
    scores = []
    for cls in range(num_classes):
        tp = sum(a == cls and b == cls for a, b in zip(y_true, y_pred))
        fp = sum(a != cls and b == cls for a, b in zip(y_true, y_pred))
        fn = sum(a == cls and b != cls for a, b in zip(y_true, y_pred))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(2 * precision * recall / max(1e-12, precision + recall))
    return sum(scores) / num_classes

def confusion_matrix(y_true, y_pred, num_classes):
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for truth, prediction in zip(y_true, y_pred):
        matrix[truth][prediction] += 1
    return matrix

def build_model(device, pretrained=True, weights_path=None, freeze_backbone=False):
    model = resnet18(weights=None)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    elif pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    if freeze_backbone:
        for parameter in model.parameters(): parameter.requires_grad = False
        for parameter in model.fc.parameters(): parameter.requires_grad = True
    return model.to(device)

def run_epoch(model, loader, loss_fn, optimizer, device, training):
    model.train(training)
    losses, truths, predictions = [], [], []
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item() * len(labels))
        truths.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return sum(losses) / len(loader.dataset), truths, predictions

def main():
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST ResNet-18 classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="skip the pretrained ResNet-18 weight download for a quick smoke test")
    parser.add_argument("--limit-per-split", type=int, default=0, help="use only this many samples per split; 0 means all")
    parser.add_argument("--fake-data", action="store_true", help="use generated random images for an offline smoke test")
    parser.add_argument("--weights", default="", help="local ResNet-18 weights file (.pth), avoids downloading")
    parser.add_argument("--no-augmentation", action="store_true", help="disable training image augmentation")
    parser.add_argument("--freeze-backbone", action="store_true", help="train only the final classifier layer")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set = make_datasets(args.data_dir, fake=args.fake_data, fake_size=args.limit_per_split or 1000, augment=not args.no_augmentation)
    if args.limit_per_split:
        train_set = torch.utils.data.Subset(train_set, range(min(args.limit_per_split, len(train_set))))
        val_set = torch.utils.data.Subset(val_set, range(min(args.limit_per_split, len(val_set))))
        test_set = torch.utils.data.Subset(test_set, range(min(args.limit_per_split, len(test_set))))
    loaders = [DataLoader(ds, args.batch_size, shuffle=i == 0, num_workers=0) for i, ds in enumerate((train_set, val_set, test_set))]
    model = build_model(device, pretrained=not args.no_pretrained, weights_path=args.weights or None, freeze_backbone=args.freeze_backbone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, y_train, p_train = run_epoch(model, loaders[0], loss_fn, optimizer, device, True)
        val_loss, y_val, p_val = run_epoch(model, loaders[1], loss_fn, optimizer, device, False)
        train_acc = accuracy_score(y_train, p_train)
        val_acc = accuracy_score(y_val, p_val)
        val_f1 = macro_f1(y_val, p_val, len(CLASSES))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_accuracy": train_acc, "val_accuracy": val_acc, "val_macro_f1": val_f1})
        print(f"epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "classes": CLASSES}, output / "best.pt")
    frame = pd.DataFrame(history)
    frame.to_csv(output / "history.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Markers keep a one-epoch run visible; plain lines have no visible
    # segment when there is only one recorded x-value.
    plot_kwargs = {"marker": "o", "linewidth": 2, "markersize": 7}
    axes[0].plot(frame.epoch, frame.train_loss, label="train", **plot_kwargs)
    axes[0].plot(frame.epoch, frame.val_loss, label="validation", **plot_kwargs)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.25); axes[0].legend()
    axes[1].plot(frame.epoch, frame.train_accuracy, label="train", **plot_kwargs)
    axes[1].plot(frame.epoch, frame.val_accuracy, label="validation", **plot_kwargs)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.25); axes[1].legend()
    if len(frame) == 1:
        for axis in axes:
            axis.set_xlim(0.75, 1.25)
    figure.tight_layout(); figure.savefig(output / "training_curves.png", dpi=160); plt.close(figure)
    checkpoint = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    _, truths, predictions = run_epoch(model, loaders[2], loss_fn, optimizer, device, False)
    test_acc = accuracy_score(truths, predictions)
    test_f1 = macro_f1(truths, predictions, len(CLASSES))
    print(f"test accuracy={test_acc:.3f} macro_f1={test_f1:.3f} device={device}")
    report_rows = []
    for index, name in enumerate(CLASSES):
        tp = sum(a == index and b == index for a, b in zip(truths, predictions))
        fp = sum(a != index and b == index for a, b in zip(truths, predictions))
        fn = sum(a == index and b != index for a, b in zip(truths, predictions))
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        report_rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1})
    pd.DataFrame(report_rows).to_csv(output / "classification_report.csv", index=False)
    pd.DataFrame({"true": [CLASSES[y] for y in truths], "pred": [CLASSES[p] for p in predictions]}).to_csv(output / "test_predictions.csv", index=False)
    matrix = confusion_matrix(truths, predictions, len(CLASSES))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(); plt.xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right"); plt.yticks(range(len(CLASSES)), CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig(output / "confusion_matrix.png", dpi=160)

if __name__ == "__main__":
    main()

