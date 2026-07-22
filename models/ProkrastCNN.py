import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


MODELS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: Path = (
        MODELS_DIR
        / "dataset"
        / "archive"
        / "Human Action Recognition"
    )
    image_size: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    epochs: int = 20
    random_seed: int = 42
    validation_fraction: float = 0.15

class ActionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        annotations_path: Path,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.images_dir = images_dir
        self.class_to_index = class_to_index
        self.transform = transform
        self.samples: list[tuple[str, str]] = []

        with annotations_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames != ["filename", "label"]:
                raise ValueError(
                    "Die CSV benötigt die Spalten filename,label"
                )

            for row in reader:
                filename = row["filename"]
                label = row["label"]

                if label not in class_to_index:
                    raise ValueError(f"Unbekanntes Label: {label}")

                image_path = images_dir / filename
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Bild aus CSV nicht gefunden: {image_path}"
                    )

                self.samples.append((filename, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        filename, label = self.samples[index]
        image_path = self.images_dir / filename

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image_tensor = self.transform(image)
        label_index = self.class_to_index[label]

        return image_tensor, torch.tensor(
            label_index,
            dtype=torch.long,
        )


def read_labels(annotations_path: Path) -> list[str]:
    with annotations_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        labels = {row["label"] for row in reader}

    return sorted(labels)

class ProkrastCNN(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


def create_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalization = transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    training_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return training_transforms, evaluation_transforms


def create_data_loaders(
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, list[str]]:
    images_dir = config.dataset_dir / "train"
    annotations_path = config.dataset_dir / "Training_set.csv"

    class_names = read_labels(annotations_path)
    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    training_transforms, evaluation_transforms = create_transforms(
        config.image_size
    )

    training_dataset = ActionDataset(
        images_dir=images_dir,
        annotations_path=annotations_path,
        class_to_index=class_to_index,
        transform=training_transforms,
    )

    validation_dataset = ActionDataset(
        images_dir=images_dir,
        annotations_path=annotations_path,
        class_to_index=class_to_index,
        transform=evaluation_transforms,
    )

    generator = torch.Generator().manual_seed(config.random_seed)
    indices = torch.randperm(
        len(training_dataset),
        generator=generator,
    ).tolist()

    validation_size = int(
        len(indices) * config.validation_fraction
    )

    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    training_subset = Subset(
        training_dataset,
        training_indices,
    )


    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    training_loader = DataLoader(
        training_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return training_loader, validation_loader, class_names


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def show_predictions(
    model: ProkrastCNN,
    validation_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    number_of_images: int = 4,
) -> None:
    random_indices = torch.randperm(
        len(validation_loader.dataset)
    )[:number_of_images]
    model.eval()
    selected_images = []
    selected_labels = []  

    for index in random_indices:
     image, label = validation_loader.dataset[index.item()]
     selected_images.append(image)
     selected_labels.append(label)

    images = torch.stack(selected_images)
    labels = torch.stack(selected_labels)

    with torch.no_grad():
        outputs = model(images.to(device, non_blocking=True))
        predictions = outputs.argmax(dim=1).cpu()

    display_images = images * 0.5 + 0.5
    figure, axes = plt.subplots(1, len(display_images), figsize=(12, 3))
    axes = np.atleast_1d(axes)

    for axis, image, label, prediction in zip(
        axes,
        display_images,
        labels,
        predictions,
    ):
        axis.imshow(image.permute(1, 2, 0).clamp(0, 1))
        axis.set_title(
            f"Echt: {class_names[label.item()]}\n"
            f"Modell: {class_names[prediction.item()]}"
        )
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def main() -> None:
    config = TrainingConfig()
    torch.manual_seed(config.random_seed)

    training_loader, validation_loader, class_names = create_data_loaders(config)
    device = select_device()
    model = ProkrastCNN(number_of_classes=len(class_names)).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    print(f"Geraet: {device}")
    print(f"Klassen: {class_names}")
    print(f"Trainings-Batches: {len(training_loader)}")
    print(f"Validation-Batches: {len(validation_loader)}")
    print(model)
    print(f"Loss: {loss_function.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    training_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    model_path = MODELS_DIR / "PoviNet.pt"

    for epoch in range(config.epochs):
        model.train()
        running_training_loss = 0.0
        correct_training_predictions = 0
        training_samples = 0

        for inputs, labels in training_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_training_loss += loss.item()
            predicted_classes = outputs.argmax(dim=1)
            correct_training_predictions += (
                predicted_classes == labels
            ).sum().item()
            training_samples += labels.size(0)

        average_training_loss = running_training_loss / len(training_loader)
        training_accuracy = correct_training_predictions / training_samples
        training_losses.append(average_training_loss)

        model.eval()
        running_validation_loss = 0.0
        correct_predictions = 0
        validation_samples = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                running_validation_loss += loss.item()

                predicted_classes = outputs.argmax(dim=1)
                correct_predictions += (
                    predicted_classes == labels
                ).sum().item()
                validation_samples += labels.size(0)

        average_validation_loss = (
            running_validation_loss / len(validation_loader)
        )
        validation_losses.append(average_validation_loss)

        validation_accuracy = correct_predictions / validation_samples
        scheduler.step(average_validation_loss)

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            torch.save(model.state_dict(), model_path)

        print(
            f"Epoch {epoch + 1:02d}/{config.epochs} | "
            f"Train Loss: {average_training_loss:.4f} | "
            f"Train Accuracy: {training_accuracy:.2%} | "
            f"Validation Loss: {average_validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.2%} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}"
        )

    print(f"Bestes Modell gespeichert: {model_path}")

    epochs = np.arange(1, config.epochs + 1)
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    show_predictions(model, validation_loader, class_names, device)

if __name__ == "__main__":
    main()
