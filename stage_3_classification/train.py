from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def train(
    dataset_dir: Path,
    output_path: Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Defect crop dataset not found: {dataset_dir}")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    dataset = datasets.ImageFolder(str(dataset_dir), transform=transform)
    if len(dataset) < 2 or len(dataset.classes) < 2:
        raise ValueError("Training requires at least two images across two defect classes")

    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    if train_size == 0 or validation_size == 0:
        raise ValueError("80/20 split requires at least two dataset images")
    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = random_split(
        dataset, [train_size, validation_size], generator=generator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                correct += int((model(images).argmax(dim=1) == labels).sum().item())
        validation_accuracy = correct / validation_size
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss / train_size:.4f}, "
            f"val_accuracy={validation_accuracy:.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "classes": dataset.classes,
            "architecture": "resnet18",
        },
        output_path,
    )
    print(f"Saved classifier checkpoint to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ResNet18 defect classifier")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/defect_crops"))
    parser.add_argument("--output", type=Path, default=Path("models/classifier.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
