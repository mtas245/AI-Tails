from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


ROOT = Path(__file__).resolve().parent
DATASET_CSV = ROOT / "data" / "dataset.csv"
MODEL_OUT = ROOT / "models" / "health_classifier.pt"
METRICS_OUT = ROOT / "models" / "health_classifier_metrics.json"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224

CLASS_NAMES = ["HEALTHY", "MONITOR", "VET_CHECK"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CatFaceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = ROOT / row["face_path"]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_INDEX[row["label"]]
        return image, label


def stratified_group_split(df: pd.DataFrame):
    rng = random.Random(SEED)

    train_cats = []
    test_cats = []

    for label in CLASS_NAMES:
        cats = sorted(df.loc[df["label"] == label, "cat_id"].unique().tolist())
        rng.shuffle(cats)

        if len(cats) < 2:
            raise ValueError(
                f"Not enough unique cats in class {label}: {len(cats)}. "
                "Need at least 2 for group split."
            )

        n_test = max(1, round(len(cats) * 0.20))
        test_cats.extend(cats[:n_test])
        train_cats.extend(cats[n_test:])

    train_df = df[df["cat_id"].isin(train_cats)].copy()
    test_df = df[df["cat_id"].isin(test_cats)].copy()

    return train_df, test_df, train_cats, test_cats


def make_class_weights(train_df: pd.DataFrame):
    counts = train_df["label"].value_counts()
    total = len(train_df)

    weights = []
    for class_name in CLASS_NAMES:
        count = int(counts.get(class_name, 0))
        if count == 0:
            raise ValueError(f"No training samples for class {class_name}")
        weights.append(total / (len(CLASS_NAMES) * count))

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
    )

    return report, matrix, all_targets, all_preds


def main():
    set_seed()

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_CSV}\n"
            "Run: python prepare_dataset.py"
        )

    df = pd.read_csv(DATASET_CSV)

    required = {"cat_id", "face_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")

    df = df[df["label"].isin(CLASS_NAMES)].copy()

    print("=== DATASET ===")
    print("Samples:", len(df))
    print("\nSamples per class:")
    print(df["label"].value_counts())
    print("\nUnique cats per class:")
    print(df.groupby("label")["cat_id"].nunique())

    train_df, test_df, train_cats, test_cats = stratified_group_split(df)

    print("\n=== GROUPED SPLIT ===")
    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))
    print("Train cats:", len(train_cats))
    print("Test cats:", len(test_cats))

    print("\nTrain unique cats per class:")
    print(train_df.groupby("label")["cat_id"].nunique())

    print("\nTest unique cats per class:")
    print(test_df.groupby("label")["cat_id"].nunique())

    overlap = set(train_df["cat_id"]) & set(test_df["cat_id"])
    if overlap:
        raise RuntimeError(f"Cat leakage detected: {overlap}")

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(
                brightness=0.12,
                contrast=0.12,
                saturation=0.08,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = CatFaceDataset(train_df, transform=train_transform)
    test_dataset = CatFaceDataset(test_df, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))
    model = model.to(device)

    class_weights = make_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_state = None

    print("\n=== TRAINING ===")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        report, _, _, _ = evaluate(model, test_loader, device)
        macro_f1 = report["macro avg"]["f1-score"]
        balanced_accuracy = np.mean(
            [report[name]["recall"] for name in CLASS_NAMES]
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"macro_f1={macro_f1:.3f} | "
            f"balanced_acc={balanced_accuracy:.3f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")

    model.load_state_dict(best_state)
    model = model.to(device)

    report, matrix, targets, preds = evaluate(model, test_loader, device)

    print("\n=== FINAL TEST RESULT ===")
    print(
        classification_report(
            targets,
            preds,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    balanced_accuracy = float(
        np.mean([report[name]["recall"] for name in CLASS_NAMES])
    )

    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": "mobilenet_v3_small",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "state_dict": {
            key: value.cpu()
            for key, value in model.state_dict().items()
        },
    }

    torch.save(checkpoint, MODEL_OUT)

    metrics = {
        "samples_total": int(len(df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_cats": len(train_cats),
        "test_cats": len(test_cats),
        "test_cat_ids": test_cats,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "balanced_accuracy": balanced_accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    METRICS_OUT.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved model:")
    print(MODEL_OUT)
    print("Saved metrics:")
    print(METRICS_OUT)

    print(
        "\nNOTE: This is a hackathon screening baseline, not a medical "
        "diagnosis model. The HEALTHY class currently contains very few "
        "unique cats, so generalization estimates remain uncertain."
    )


if __name__ == "__main__":
    main()
