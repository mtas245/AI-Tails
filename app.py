from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
YOLO_MODEL = ROOT / "models" / "cat_face_yolov8n.pt"
CLASSIFIER_MODEL = ROOT / "models" / "health_classifier.pt"

CLASS_LABELS_DE = {
    "HEALTHY": "Geringes Risiko",
    "MONITOR": "Beobachten",
    "VET_CHECK": "Tierarzt empfohlen",
}

CLASS_EMOJI = {
    "HEALTHY": "🟢",
    "MONITOR": "🟡",
    "VET_CHECK": "🔴",
}


@st.cache_resource
def load_models():
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(f"YOLO model missing: {YOLO_MODEL}")
    if not CLASSIFIER_MODEL.exists():
        raise FileNotFoundError(
            f"Classifier missing: {CLASSIFIER_MODEL}\n"
            "Run: python train_model.py"
        )

    detector = YOLO(str(YOLO_MODEL))

    checkpoint = torch.load(
        CLASSIFIER_MODEL,
        map_location="cpu",
        weights_only=False,
    )

    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))

    classifier = mobilenet_v3_small(weights=None)
    in_features = classifier.classifier[3].in_features
    classifier.classifier[3] = nn.Linear(in_features, len(class_names))
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()

    return detector, classifier, class_names, image_size


def classifier_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def crop_face_from_result(result, image_bgr):
    if result is None or len(result.boxes) == 0:
        return None, None

    boxes = result.boxes
    coords = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
    index = int(np.argmax(areas))

    x1, y1, x2, y2 = map(int, coords[index])
    height, width = image_bgr.shape[:2]

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

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    return crop, float(confidences[index])


def detect_face(detector, image_bgr):
    results = detector.predict(
        source=image_bgr,
        conf=0.25,
        imgsz=416,
        verbose=False,
    )

    if not results:
        return None, None

    return crop_face_from_result(results[0], image_bgr)


def predict_faces_batch(classifier, transform, faces_bgr):
    tensors = []

    for face_bgr in faces_bgr:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensors.append(transform(image))

    batch = torch.stack(tensors)

    with torch.inference_mode():
        logits = classifier(batch)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    return probabilities


def predict_face(classifier, class_names, transform, face_bgr):
    return predict_faces_batch(
        classifier,
        transform,
        [face_bgr],
    )[0]


def analyze_image(detector, classifier, class_names, transform, image_bgr):
    face, detector_conf = detect_face(detector, image_bgr)
    if face is None:
        return None

    probabilities = predict_face(
        classifier,
        class_names,
        transform,
        face,
    )

    return {
        "face": face,
        "detector_confidence": detector_conf,
        "probabilities": probabilities,
    }


def analyze_video(
    detector,
    classifier,
    class_names,
    transform,
    video_path: str,
    max_samples: int = 12,
):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return []

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 25.0

    if frame_count <= 0:
        capture.release()
        return []

    sample_count = min(max_samples, frame_count)
    sample_indices = np.linspace(
        0,
        frame_count - 1,
        num=sample_count,
        dtype=int,
    )

    frames = []
    metadata = []

    try:
        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue

            frames.append(frame)
            metadata.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp": float(frame_index / fps),
                }
            )
    finally:
        capture.release()

    if not frames:
        return []

    results = detector.predict(
        source=frames,
        conf=0.25,
        imgsz=416,
        verbose=False,
    )

    faces = []
    face_meta = []

    for frame, result, meta in zip(frames, results, metadata):
        face, detector_conf = crop_face_from_result(result, frame)
        if face is None:
            continue

        faces.append(face)
        face_meta.append(
            {
                **meta,
                "face": face,
                "detector_confidence": detector_conf,
            }
        )

    if not faces:
        return []

    probabilities_batch = predict_faces_batch(
        classifier,
        transform,
        faces,
    )

    analyses = []

    for meta, probabilities in zip(face_meta, probabilities_batch):
        analyses.append(
            {
                **meta,
                "probabilities": probabilities,
            }
        )

    return analyses


def estimate_fgs_proxy(class_names, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)

    anchors = {
        "HEALTHY": 1.5,
        "MONITOR": 5.5,
        "VET_CHECK": 9.0,
    }

    score = 0.0
    for class_name, probability in zip(class_names, probabilities):
        score += anchors[class_name] * float(probability)

    return float(np.clip(score, 0.0, 10.0))


def show_ai_result(class_names, probabilities, face_count: int):
    probabilities = np.asarray(probabilities, dtype=float)
    winner_index = int(np.argmax(probabilities))
    winner = class_names[winner_index]
    confidence = float(probabilities[winner_index])
    fgs_proxy = estimate_fgs_proxy(class_names, probabilities)
    proxy_interpretation, proxy_emoji = fgs_interpretation(int(round(fgs_proxy)))

    st.subheader("KI-Einschätzung")
    st.markdown(
        f"## {CLASS_EMOJI[winner]} "
        f"{CLASS_LABELS_DE[winner]}"
    )

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("KI-Konfidenz", f"{confidence * 100:.1f}%")
    with metric_col2:
        st.metric("Automatischer FGS-Proxy", f"{fgs_proxy:.1f}/10")

    st.caption(f"Ausgewertete Katzengesichter: {face_count}")
    st.markdown(f"**{proxy_emoji} Proxy-Interpretation:** {proxy_interpretation}")
    st.caption(
        "Der automatische FGS-Proxy wird aus den Modellwahrscheinlichkeiten "
        "abgeleitet. Er ist noch kein echter FGS-Score der fünf einzelnen Action Units."
    )

    st.write("Klassenwahrscheinlichkeiten")
    for class_name, probability in zip(class_names, probabilities):
        st.progress(
            float(probability),
            text=f"{CLASS_LABELS_DE[class_name]}: {probability * 100:.1f}%",
        )

    if face_count < 3:
        st.warning(
            "Nur wenig visuelle Evidenz verfügbar. "
            "Für eine stabilere Einschätzung sollte ein längeres Video "
            "oder mehrere gute Aufnahmen verwendet werden."
        )

    return winner, confidence, fgs_proxy


def fgs_interpretation(score: int):
    if score == 0:
        return "Kein Schmerz erkennbar", "🟢"
    if score <= 3:
        return "Kein oder leichter Schmerz – regelmäßig neu beurteilen", "🟡"
    if score <= 8:
        return "Schmerz wahrscheinlich – tierärztliche Beurteilung empfohlen", "🟠"
    return "Starker Schmerz wahrscheinlich – tierärztliche Beurteilung empfohlen", "🔴"


def show_manual_fgs():
    st.divider()
    st.subheader("Feline Grimace Scale (FGS)")

    st.caption(
        "Die FGS bewertet fünf Action Units jeweils von 0 bis 2. "
        "Diese Werte werden hier bewusst nicht automatisch erfunden, "
        "da unser aktueller Trainingsdatensatz noch keine Action-Unit-Labels enthält."
    )

    col1, col2 = st.columns(2)

    with col1:
        ears = st.select_slider(
            "Ohrenposition",
            options=[0, 1, 2],
            value=0,
            help="0 = nach vorne, 1 = leicht auseinander, 2 = nach außen rotiert",
        )
        eyes = st.select_slider(
            "Augenverengung",
            options=[0, 1, 2],
            value=0,
            help="0 = offen, 1 = teilweise geschlossen, 2 = zusammengekniffen",
        )
        muzzle = st.select_slider(
            "Muzzle-/Maulspannung",
            options=[0, 1, 2],
            value=0,
            help="0 = entspannt/rund, 1 = leichte Spannung, 2 = angespannt/elliptisch",
        )

    with col2:
        whiskers = st.select_slider(
            "Schnurrhaare",
            options=[0, 1, 2],
            value=0,
            help="0 = locker/gebogen, 1 = gerader/näher zusammen, 2 = gerade nach vorne",
        )
        head = st.select_slider(
            "Kopfposition",
            options=[0, 1, 2],
            value=0,
            help="0 = über Schulterlinie, 1 = auf Schulterlinie, 2 = darunter/gesenkt",
        )

    total = ears + eyes + muzzle + whiskers + head
    interpretation, emoji = fgs_interpretation(total)

    st.metric("FGS-Gesamtscore", f"{total}/10")
    st.markdown(f"### {emoji} {interpretation}")

    st.caption(
        "FGS wurde für akute Schmerzen entwickelt. "
        "Stress, Angst, Sedierung und chronische Erkrankungen können die "
        "Interpretation beeinflussen."
    )


def main():
    st.set_page_config(
        page_title="AI-Tails FGS Screening",
        page_icon="🐱",
        layout="wide",
    )

    st.title("🐱 AI-Tails – FGS Screening Demo")
    st.write(
        "Kamera-basierte Einschätzung von schmerzbezogenen "
        "Gesichtsveränderungen bei Katzen."
    )

    st.info(
        "Proof of Concept für den Makeathon: Das System ist ein Screening-Werkzeug "
        "und keine tierärztliche Diagnose."
    )

    try:
        detector, classifier, class_names, image_size = load_models()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    transform = classifier_transform(image_size)

    uploaded = st.file_uploader(
        "Katzenbild oder Video hochladen",
        type=["jpg", "jpeg", "png", "mp4", "mov", "m4v", "avi"],
    )

    if uploaded is None:
        show_manual_fgs()
        return

    suffix = Path(uploaded.name).suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png"}:
        image_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Bild konnte nicht gelesen werden.")
            return

        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption="Originalaufnahme",
            use_container_width=True,
        )

        result = analyze_image(
            detector,
            classifier,
            class_names,
            transform,
            image_bgr,
        )

        if result is None:
            st.error("Kein Katzengesicht erkannt.")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(
                    cv2.cvtColor(result["face"], cv2.COLOR_BGR2RGB),
                    caption=(
                        "Erkanntes Katzengesicht "
                        f"(YOLO {result['detector_confidence'] * 100:.1f}%)"
                    ),
                    use_container_width=True,
                )

            with col2:
                show_ai_result(
                    class_names,
                    result["probabilities"],
                    face_count=1,
                )

    else:
        st.video(uploaded)

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp:
            temp.write(uploaded.getbuffer())
            temp_path = temp.name

        with st.spinner("Video wird analysiert …"):
            analyses = analyze_video(
                detector,
                classifier,
                class_names,
                transform,
                temp_path,
            )

        if not analyses:
            st.error("Im Video wurde kein auswertbares Katzengesicht erkannt.")
        else:
            probabilities = np.mean(
                [item["probabilities"] for item in analyses],
                axis=0,
            )

            best_item = max(
                analyses,
                key=lambda item: item["detector_confidence"],
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(
                    cv2.cvtColor(
                        best_item["face"],
                        cv2.COLOR_BGR2RGB,
                    ),
                    caption=(
                        "Bestes erkanntes Gesicht "
                        f"bei {best_item['timestamp']:.1f} s"
                    ),
                    use_container_width=True,
                )

            with col2:
                show_ai_result(
                    class_names,
                    probabilities,
                    face_count=len(analyses),
                )

    show_manual_fgs()

    st.divider()
    st.caption(
        "FGS-orientierter Forschungsprototyp. "
        "Die automatische KI-Klasse basiert aktuell auf einem kleinen "
        "Trainingsdatensatz und ersetzt keine klinische Untersuchung."
    )


if __name__ == "__main__":
    main()
