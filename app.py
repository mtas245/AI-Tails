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
from fgs_auto import estimate_fgs, DISPLAY_NAMES


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


def bgr_to_pil(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


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


def show_auto_fgs_result(faces_bgr):
    st.subheader("Automatische FGS-Bewertung")

    face_images = [bgr_to_pil(face) for face in faces_bgr]

    try:
        result = estimate_fgs(face_images)
    except Exception as exc:
        st.warning(
            "Automatische FGS-Bewertung konnte nicht geladen werden. "
            "Installiere ggf. transformers: python -m pip install transformers"
        )
        st.caption(str(exc))
        return

    action_units = result["action_units"]
    total = result["total"]
    continuous_total = result["continuous_total"]

    cols = st.columns(5)

    unit_order = ["ears", "eyes", "muzzle", "whiskers", "head"]

    for column, unit in zip(cols, unit_order):
        value = action_units[unit]["score"]
        expected = action_units[unit]["expected_score"]

        with column:
            st.metric(
                DISPLAY_NAMES[unit],
                f"{value}/2",
                help=f"Kontinuierliche ML-Schätzung: {expected:.2f}",
            )

    interpretation, emoji = fgs_interpretation(total)

    st.metric("Automatischer FGS-Gesamtscore", f"{total}/10")
    st.markdown(f"### {emoji} {interpretation}")

    st.caption(
        f"Kontinuierlicher ML-Score: {continuous_total:.2f}/10. "
        "Dieser Wert wird automatisch aus einem vortrainierten Vision-Modell "
        "abgeleitet und ist noch nicht klinisch validiert."
    )


def show_manual_fgs():
    with st.expander("Manuelle FGS-Referenz anzeigen"):
        st.caption(
            "Optionaler Referenzbereich: Hier kannst du die fünf FGS-Action-Units "
            "manuell mit 0 bis 2 bewerten. Dieser manuelle Score ist getrennt von "
            "der automatischen KI-Session-Auswertung."
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

        st.metric("Manueller FGS-Gesamtscore", f"{total}/10")
        st.markdown(f"### {emoji} {interpretation}")

        st.caption(
            "Dieser Bereich ist nur eine manuelle FGS-Referenz und wird nicht "
            "automatisch aus den hochgeladenen Dateien berechnet."
        )


def main():
    st.set_page_config(
        page_title="AI-Tails FGS Screening",
        page_icon="🐱",
        layout="wide",
    )

    st.title("🐱 AI-Tails – FGS Screening Demo")
    st.write(
        "Mehrere Bilder und Videos derselben Katze werden gemeinsam als "
        "eine Untersuchungssession ausgewertet."
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

    uploads = st.file_uploader(
        "Bilder und/oder Videos derselben Katze hochladen",
        type=["jpg", "jpeg", "png", "mp4", "mov", "m4v", "avi"],
        accept_multiple_files=True,
        help="Du kannst mehrere Dateien derselben Katze gleichzeitig auswählen.",
    )

    if not uploads:
        st.caption(
            "Noch keine Session geladen. Wähle mehrere Bilder oder Videos "
            "derselben Katze aus."
        )
        show_manual_fgs()
        return

    all_analyses = []
    failed_files = []
    preview_faces = []

    with st.spinner(f"{len(uploads)} Datei(en) werden gemeinsam analysiert …"):
        for uploaded in uploads:
            suffix = Path(uploaded.name).suffix.lower()

            if suffix in {".jpg", ".jpeg", ".png"}:
                image_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
                image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                if image_bgr is None:
                    failed_files.append(uploaded.name)
                    continue

                result = analyze_image(
                    detector,
                    classifier,
                    class_names,
                    transform,
                    image_bgr,
                )

                if result is None:
                    failed_files.append(uploaded.name)
                    continue

                result["source_name"] = uploaded.name
                result["source_type"] = "image"
                all_analyses.append(result)
                preview_faces.append(result)

            else:
                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as temp:
                    temp.write(uploaded.getbuffer())
                    temp_path = temp.name

                analyses = analyze_video(
                    detector,
                    classifier,
                    class_names,
                    transform,
                    temp_path,
                )

                if not analyses:
                    failed_files.append(uploaded.name)
                    continue

                for item in analyses:
                    item["source_name"] = uploaded.name
                    item["source_type"] = "video"
                    all_analyses.append(item)

                preview_faces.append(
                    max(
                        analyses,
                        key=lambda item: item["detector_confidence"],
                    )
                )

    st.subheader("Session-Zusammenfassung")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Dateien", len(uploads))
    col_b.metric("Auswertbare Gesichter", len(all_analyses))
    col_c.metric("Nicht auswertbare Dateien", len(failed_files))

    if not all_analyses:
        st.error(
            "In den hochgeladenen Dateien wurde kein auswertbares Katzengesicht erkannt."
        )
        if failed_files:
            st.write("Nicht auswertbar:", ", ".join(failed_files))
        show_manual_fgs()
        return

    session_probabilities = np.mean(
        [item["probabilities"] for item in all_analyses],
        axis=0,
    )

    left, right = st.columns([1, 1])

    with left:
        best_item = max(
            all_analyses,
            key=lambda item: item["detector_confidence"],
        )
        st.image(
            cv2.cvtColor(best_item["face"], cv2.COLOR_BGR2RGB),
            caption=(
                f"Bestes erkanntes Gesicht aus {best_item['source_name']} "
                f"(YOLO {best_item['detector_confidence'] * 100:.1f}%)"
            ),
            use_container_width=True,
        )

    with right:
        show_ai_result(
            class_names,
            session_probabilities,
            face_count=len(all_analyses),
        )

    show_auto_fgs_result(
        [item["face"] for item in all_analyses[:12]]
    )

    if len(preview_faces) > 1:
        st.subheader("Erkannte Gesichter aus der Session")
        columns = st.columns(min(4, len(preview_faces)))

        for index, item in enumerate(preview_faces[:8]):
            with columns[index % len(columns)]:
                st.image(
                    cv2.cvtColor(item["face"], cv2.COLOR_BGR2RGB),
                    caption=item["source_name"],
                    use_container_width=True,
                )

    if failed_files:
        with st.expander("Nicht auswertbare Dateien"):
            for name in failed_files:
                st.write(f"- {name}")

    show_manual_fgs()

    st.divider()
    st.caption(
        "Automatische Session-Einschätzung = Mittelwert über alle erkannten "
        "Gesichter der hochgeladenen Bilder und Video-Frames. "
        "Der FGS-Proxy ist eine KI-Näherung und noch kein automatisch "
        "bestimmter echter FGS-Action-Unit-Score."
    )


if __name__ == "__main__":
    main()
