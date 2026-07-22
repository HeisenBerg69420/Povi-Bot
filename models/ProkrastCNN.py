"""Kleines CNN-Trainingsgeruest fuer die lokale Povi-Bot-Vision-Pipeline.

Erwartete Ordnerstruktur:

models/dataset/
    train/
        no-person/
        person/
    validation/
        no-person/
        person/

Start im Repository-Root:
    python models/ProkrastCNN.py
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


MODELS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: Path = MODELS_DIR / "dataset"
    image_size: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    random_seed: int = 42


class ProkrastCNN(nn.Module):
    """Kleiner Bildklassifikator fuer zwei oder mehr Klassen."""

    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, number_of_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened)


def create_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Erzeugt getrennte Transformationen fuer Training und Auswertung."""

    training_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),  # RGB [0, 255] -> NCHW float32 [0, 1]
        ]
    )
    evaluation_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return training_transforms, evaluation_transforms


def create_data_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, list[str]]:
    training_dir = config.dataset_dir / "train"
    validation_dir = config.dataset_dir / "validation"
    if not training_dir.is_dir() or not validation_dir.is_dir():
        raise FileNotFoundError(
            "Datensatz fehlt. Lege models/dataset/train und "
            "models/dataset/validation mit je einem Unterordner pro Klasse an."
        )

    training_transforms, evaluation_transforms = create_transforms(config.image_size)
    training_dataset = datasets.ImageFolder(training_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(validation_dir, transform=evaluation_transforms)

    if training_dataset.classes != validation_dataset.classes:
        raise ValueError("Training und Validation muessen dieselben Klassenordner enthalten.")

    generator = torch.Generator().manual_seed(config.random_seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    return training_loader, validation_loader, training_dataset.classes


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    config = TrainingConfig()
    torch.manual_seed(config.random_seed)

    training_loader, validation_loader, classes = create_data_loaders(config)
    device = select_device()
    model = ProkrastCNN(number_of_classes=len(classes)).to(device)

    # Diese beiden Objekte brauchst du im naechsten Schritt im Trainingsloop.
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Geraet: {device}")
    print(f"Klassen: {classes}")
    print(f"Trainings-Batches: {len(training_loader)}")
    print(f"Validation-Batches: {len(validation_loader)}")
    print(model)
    print(f"Loss: {loss_function.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print("Naechster Schritt: Trainings- und Validation-Loop implementieren.")


if __name__ == "__main__":
    main()
