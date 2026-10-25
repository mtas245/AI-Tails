from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd

from fgs_geometry import extract_geometry_features, fit_healthy_baseline
from landmark_detector import predict_landmarks


ROOT = Path(__file__).resolve().parent
DATASET_CSV = ROOT / "data" / "dataset.csv"
BASELINE_OUT = ROOT / "models" / "healthy_geometry_baseline.json"

MAX_IMAGES_PER_CAT = 20


def main():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATASET_CSV}\n"
            "Run prepare_dataset.py first."
        )

    df = pd.read_csv(DATASET_CSV)

    required = {"cat_id", "face_path", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in dataset.csv: {sorted(missing)}"
        )

    healthy = df[df["label"] == "HEALTHY"].copy()

    if healthy.empty:
        raise ValueError("No HEALTHY samples found.")

    print("=== HEALTHY GEOMETRY BASELINE ===")
    print("Healthy images:", len(healthy))
    print("Healthy cats:", healthy["cat_id"].nunique())

    cat_feature_rows = []
    failed = 0
    processed = 0

    for cat_id, group in healthy.groupby("cat_id"):
        group = group.head(MAX_IMAGES_PER_CAT)
        per_image_features = []

        print(f"Processing {cat_id}: {len(group)} images")

        for _, row in group.iterrows():
            face_path = ROOT / str(row["face_path"])
            image = cv2.imread(str(face_path))

            if image is None:
                failed += 1
                continue

            try:
                landmarks = predict_landmarks(image)
                features = extract_geometry_features(landmarks)
                per_image_features.append(features)
                processed += 1

            except Exception as exc:
                failed += 1
                print(f"  ERROR {face_path.name}: {exc}")

        if not per_image_features:
            continue

        keys = per_image_features[0].keys()

        # One reference row per cat: median over that cat's images.
        cat_features = {
            key: float(
                np.median([item[key] for item in per_image_features])
            )
            for key in keys
        }

        cat_feature_rows.append(cat_features)

    if len(cat_feature_rows) < 3:
        raise RuntimeError(
            "Need at least 3 healthy cats with usable landmarks "
            f"for a baseline, got {len(cat_feature_rows)}."
        )

    baseline = fit_healthy_baseline(cat_feature_rows)

    payload = {
        "healthy_cats": len(cat_feature_rows),
        "processed_images": processed,
        "failed_images": failed,
        "max_images_per_cat": MAX_IMAGES_PER_CAT,
        "mean": baseline.mean,
        "std": baseline.std,
        "note": (
            "Hackathon baseline built from per-cat median geometry of "
            "existing HEALTHY-labeled cats. Thresholds are not clinically validated."
        ),
    }

    BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_OUT.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nBaseline saved:")
    print(BASELINE_OUT)

    print("\nHealthy cats used:", len(cat_feature_rows))
    print("Images processed:", processed)
    print("Images failed:", failed)

    print("\nFeature means:")
    for key, value in baseline.mean.items():
        print(f"  {key}: {value:.4f}")

    print(
        "\nIMPORTANT: This is a project-specific geometric reference, "
        "not a clinically validated FGS threshold set."
    )


if __name__ == "__main__":
    main()
