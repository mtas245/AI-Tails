from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import hashlib
import re
import unicodedata

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
FACES_DIR = ROOT / "data" / "faces"
LABELS_FILE = ROOT / "data" / "labels" / "all_cats.xlsx"
MODEL_FILE = ROOT / "models" / "cat_face_yolov8n.pt"
DATASET_CSV = ROOT / "data" / "dataset.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

FACE_CONFIDENCE = 0.25
VIDEO_SAMPLE_SECONDS = 1.0
MIN_FACE_SIZE = 64

CAT_COLUMN = "CatID/Name"
PAIN_COLUMN = "Pain level according to the Glasgow scale"


def normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", "", name)
    name = re.sub(r"\([^)]*\)", "", name)

    for extra in ("kitten", "chaton", "ragdoll", "white british lh"):
        name = name.replace(extra, "")

    name = re.sub(r"\s+\d+$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def extract_pain_score(value):
    if pd.isna(value):
        return None

    match = re.match(r"\s*(\d+)", str(value))
    return int(match.group(1)) if match else None


def score_to_label(score: int | None) -> str | None:
    if score is None:
        return None
    if score <= 3:
        return "HEALTHY"
    if score <= 9:
        return "MONITOR"
    return "VET_CHECK"


def stable_cat_id(name: str) -> str:
    digest = hashlib.sha1(normalize_name(name).encode("utf-8")).hexdigest()[:8]
    return f"CAT_{digest.upper()}"


def load_cat_labels():
    df = pd.read_excel(LABELS_FILE, sheet_name="Data")

    missing = [c for c in (CAT_COLUMN, PAIN_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Excel columns: {missing}")

    df = df.copy()
    df["normalized_name"] = df[CAT_COLUMN].apply(normalize_name)
    df["pain_score"] = df[PAIN_COLUMN].apply(extract_pain_score)
    df["label"] = df["pain_score"].apply(score_to_label)

    labels = {}

    # MVP rule for duplicate Excel rows:
    # use the highest recorded pain score for the cat.
    # Later we can replace this with visit/date-specific labels.
    for norm_name, group in df.groupby("normalized_name", dropna=False):
        if not norm_name:
            continue

        valid = group.dropna(subset=["pain_score"])
        if valid.empty:
            continue

        row = valid.loc[valid["pain_score"].idxmax()]
        excel_name = str(row[CAT_COLUMN])

        labels[norm_name] = {
            "cat_id": stable_cat_id(excel_name),
            "excel_name": excel_name,
            "pain_score": int(row["pain_score"]),
            "label": str(row["label"]),
        }

    return labels


def read_image(path: Path):
    if path.suffix.lower() == ".heic":
        if not HEIC_AVAILABLE:
            raise RuntimeError(
                "HEIC detected, but pillow-heif is missing. "
                "Install it with: python -m pip install pillow-heif"
            )

        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return cv2.imread(str(path))


def detect_best_face(model: YOLO, image):
    results = model.predict(
        source=image,
        conf=FACE_CONFIDENCE,
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes
    coords = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
    index = int(np.argmax(areas))

    x1, y1, x2, y2 = map(int, coords[index])
    height, width = image.shape[:2]

    # Small padding so ears / chin are less likely to be cut off.
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = int(box_width * 0.10)
    pad_y = int(box_height * 0.10)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = image[y1:y2, x1:x2]

    if crop.shape[0] < MIN_FACE_SIZE or crop.shape[1] < MIN_FACE_SIZE:
        return None, None

    return crop, float(confidences[index])


def iter_video_frames(path: Path):
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps * VIDEO_SAMPLE_SECONDS)))
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % step == 0:
                yield frame, frame_index, frame_index / fps

            frame_index += 1
    finally:
        capture.release()


def save_face(crop, cat_id: str, source_path: Path, suffix: str) -> Path:
    target_dir = FACES_DIR / cat_id
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_path.stem)
    target = target_dir / f"{safe_stem}_{suffix}.jpg"

    if not cv2.imwrite(str(target), crop):
        raise RuntimeError(f"Could not save crop: {target}")

    return target


def main():
    for path in (RAW_DIR, LABELS_FILE, MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    FACES_DIR.mkdir(parents=True, exist_ok=True)

    labels = load_cat_labels()
    model = YOLO(str(MODEL_FILE))

    rows = []
    stats = defaultdict(int)
    unmatched = []

    cat_folders = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    print(f"Cat folders found: {len(cat_folders)}")

    for folder in sorted(cat_folders, key=lambda p: p.name.lower()):
        normalized = normalize_name(folder.name)
        info = labels.get(normalized)

        if info is None:
            unmatched.append(folder.name)
            stats["unmatched_folders"] += 1
            continue

        print(
            f"Processing {folder.name} -> "
            f"{info['label']} (pain={info['pain_score']})"
        )

        files = [p for p in folder.rglob("*") if p.is_file()]

        for media in files:
            ext = media.suffix.lower()

            if ext in IMAGE_EXTENSIONS:
                stats["images_seen"] += 1

                try:
                    image = read_image(media)
                    if image is None:
                        stats["image_read_errors"] += 1
                        continue

                    crop, confidence = detect_best_face(model, image)
                    if crop is None:
                        stats["no_face"] += 1
                        continue

                    face_path = save_face(
                        crop,
                        info["cat_id"],
                        media,
                        "image",
                    )

                    rows.append(
                        {
                            "cat_id": info["cat_id"],
                            "source_folder": folder.name,
                            "source_file": str(media.relative_to(ROOT)),
                            "media_type": "image",
                            "frame_index": None,
                            "timestamp_sec": None,
                            "face_path": str(face_path.relative_to(ROOT)),
                            "face_confidence": round(confidence, 4),
                            "pain_score": info["pain_score"],
                            "label": info["label"],
                        }
                    )
                    stats["faces_saved"] += 1

                except Exception as exc:
                    stats["errors"] += 1
                    print(f"  ERROR image {media.name}: {exc}")

            elif ext in VIDEO_EXTENSIONS:
                stats["videos_seen"] += 1

                try:
                    for frame, frame_index, timestamp in iter_video_frames(media):
                        stats["video_frames_sampled"] += 1

                        crop, confidence = detect_best_face(model, frame)
                        if crop is None:
                            stats["no_face"] += 1
                            continue

                        face_path = save_face(
                            crop,
                            info["cat_id"],
                            media,
                            f"frame_{frame_index:06d}",
                        )

                        rows.append(
                            {
                                "cat_id": info["cat_id"],
                                "source_folder": folder.name,
                                "source_file": str(media.relative_to(ROOT)),
                                "media_type": "video_frame",
                                "frame_index": frame_index,
                                "timestamp_sec": round(timestamp, 2),
                                "face_path": str(face_path.relative_to(ROOT)),
                                "face_confidence": round(confidence, 4),
                                "pain_score": info["pain_score"],
                                "label": info["label"],
                            }
                        )
                        stats["faces_saved"] += 1

                except Exception as exc:
                    stats["errors"] += 1
                    print(f"  ERROR video {media.name}: {exc}")

    dataset = pd.DataFrame(rows)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_CSV, index=False)

    print("\n=== DATASET PREPARATION COMPLETE ===")
    print(f"Rows: {len(dataset)}")
    print(f"CSV: {DATASET_CSV}")

    if not dataset.empty:
        print("\nSamples per label:")
        print(dataset["label"].value_counts())

        print("\nUnique cats per label:")
        print(dataset.groupby("label")["cat_id"].nunique())

    print("\nStats:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")

    if unmatched:
        print("\nUnmatched folders (not used):")
        for name in unmatched:
            print(f"  - {name}")

    print(
        "\nIMPORTANT: train/test splitting must be grouped by cat_id. "
        "Never split individual images from the same cat across train and test."
    )


if __name__ == "__main__":
    main()
