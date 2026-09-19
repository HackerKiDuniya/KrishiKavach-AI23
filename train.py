"""Train KrishiKavach from labelled crop images in dataset/."""

from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import get_crop_model

DATASET_DIR = Path("dataset")
MODEL_PATH = "crop_disease_model.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
SEED = 42
# Keeps first-time CPU training practical. Set to None for a full-dataset run.
MAX_IMAGES_PER_CLASS = 150


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "dataset/ folder not found. Create folders such as dataset/Apple___Healthy/ and add images."
        )
    if not any(path.is_dir() for path in DATASET_DIR.iterdir()):
        raise ValueError(
            "dataset/ has no class folders. Add folders such as Apple___Healthy/ with labelled images."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
    if len(dataset) < 2 or len(dataset.classes) < 2:
        raise ValueError("Add images for at least two labelled disease folders before training.")

    # Build a balanced starter subset so every disease category is represented equally.
    # This avoids one large class dominating a quick CPU training demonstration.
    indices = []
    rng = random.Random(SEED)
    for class_index in range(len(dataset.classes)):
        class_indices = [index for index, target in enumerate(dataset.targets) if target == class_index]
        rng.shuffle(class_indices)
        if MAX_IMAGES_PER_CLASS is not None:
            class_indices = class_indices[:MAX_IMAGES_PER_CLASS]
        indices.extend(class_indices)
    rng.shuffle(indices)
    split = max(1, int(0.8 * len(indices)))
    split = min(split, len(indices) - 1)
    train_data = Subset(dataset, indices[:split])
    valid_data = Subset(dataset, indices[split:])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_crop_model(len(dataset.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    print(
        f"Classes: {dataset.classes}\n"
        f"Images available: {len(dataset)} | Images selected: {len(indices)} | Device: {device}"
    )

    best_accuracy = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        valid_correct = valid_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images.to(device))
                valid_correct += (outputs.argmax(1).cpu() == labels).sum().item()
                valid_total += labels.size(0)
        validation_accuracy = 100 * valid_correct / valid_total
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {running_loss / total:.4f} | "
              f"Train accuracy: {100 * correct / total:.2f}% | "
              f"Validation accuracy: {validation_accuracy:.2f}%")

        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": dataset.classes,
                "num_classes": len(dataset.classes),
                "image_size": IMAGE_SIZE,
                "best_validation_accuracy": best_accuracy,
            }, MODEL_PATH)

    print(f"Training complete. Best model saved to {MODEL_PATH} ({best_accuracy:.2f}% validation accuracy).")


if __name__ == "__main__":
    main()
