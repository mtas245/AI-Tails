from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from fgs_geometry import HealthyBaseline, extract_geometry_features, score_fgs_geometry
from landmark_detector import predict_landmarks, draw_landmarks
from vision_contract import build_vision_payload


ROOT = Path(__file__).resolve().parent
YOLO_MODEL = ROOT / "models" / "cat_face_yolov8n.pt"
HEALTHY_BASELINE_FILE = ROOT / "models" / "healthy_geometry_baseline.json"


st.set_page_config(
    page_title="AI-Tails Live",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {
    max-width: 760px;
    padding-top: 1.2rem;
    padding-bottom: 4rem;
}
.hero {
    padding: 1.2rem 1.25rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(35,108,168,.22), rgba(117,66,184,.18));
    border: 1px solid rgba(255,255,255,.08);
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2rem;
    margin: 0 0 .35rem 0;
}
.hero p {
    opacity: .82;
    margin: 0;
}
.result-card {
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 20px;
    padding: 1rem 1.1rem;
    margin: .8rem 0 1rem 0;
    background: rgba(255,255,255,.03);
}
.small-muted {
    opacity: .68;
    font-size: .9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector():
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(
            "Cat-face detector missing. Expected: models/cat_face_yolov8n.pt"
        )
    return YOLO(str(YOLO_MODEL))


@st.cache_resource
def load_baseline():
    if not HEALTHY_BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Healthy baseline missing. Run: python build_healthy_baseline.py"
        )

    payload = json.loads(
        HEALTHY_BASELINE_FILE.read_text(encoding="utf-8")
    )

    baseline = HealthyBaseline(
        mean={k: float(v) for k, v in payload["mean"].items()},
        std={k: float(v) for k, v in payload["std"].items()},
    )
    return baseline, payload


def decode_image(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def crop_face(detector, image_bgr):
    results = detector.predict(
        source=image_bgr,
        conf=0.25,
        imgsz=416,
        verbose=False,
    )

    if not results or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes
    coords = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    areas = (coords[:, 2] - coords[:, 0]) * (coords[:, 3] - coords[:, 1])
    idx = int(np.argmax(areas))

    x1, y1, x2, y2 = map(int, coords[idx])
    h, w = image_bgr.shape[:2]

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * 0.10)
    pad_y = int(bh * 0.10)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, float(confs[idx])


def face_quality(face_bgr, detector_confidence):
    h, w = face_bgr.shape[:2]
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons = []

    if detector_confidence < 0.50:
        reasons.append("Gesicht wurde nur unsicher erkannt")
    if min(h, w) < 80:
        reasons.append("Gesicht ist zu klein im Bild")
    if sharpness < 30:
        reasons.append("Bild ist zu unscharf")

    return {
        "accepted": not reasons,
        "sharpness": sharpness,
        "width": w,
        "height": h,
        "reasons": reasons,
    }


def human_summary(normalized_score: float):
    if normalized_score < 3.5:
        return (
            "Gesichtssignale weitgehend unauffällig",
            "Die gemessenen Merkmale liegen überwiegend nahe an unserer gesunden Referenz.",
            "🟢",
        )
    if normalized_score < 6.5:
        return (
            "Mehrere Gesichtssignale auffällig",
            "Einige FGS-orientierte Merkmale weichen von unserer gesunden Referenz ab.",
            "🟡",
        )
    return (
        "Deutliche Abweichungen erkannt",
        "Mehrere Gesichtsmerkmale unterscheiden sich klar von unserer gesunden Referenz.",
        "🟠",
    )


def feature_explanation(key: str, score: int | None):
    texts = {
        "ears": {
            0: "Die Ohrengeometrie liegt nahe an der gesunden Referenz.",
            1: "Die Ohren stehen etwas anders als bei der gesunden Referenz.",
            2: "Die Ohrenposition weicht deutlich von der gesunden Referenz ab.",
        },
        "eyes": {
            0: "Die Augenöffnung liegt nahe an der gesunden Referenz.",
            1: "Die Augen wirken etwas stärker verengt als die gesunde Referenz.",
            2: "Die Augen wirken deutlich stärker verengt als die gesunde Referenz.",
        },
        "muzzle": {
            0: "Die Geometrie der Schnauze wirkt weitgehend entspannt.",
            1: "Im Muzzle-/Maulbereich gibt es eine moderate geometrische Abweichung.",
            2: "Im Muzzle-/Maulbereich gibt es eine deutliche geometrische Abweichung.",
        },
        "whiskers": {
            0: "Die Whisker-/Muzzle-Region liegt nahe an der Referenz.",
            1: "Die Whisker-/Muzzle-Region zeigt eine moderate Abweichung.",
            2: "Die Whisker-/Muzzle-Region zeigt eine deutliche Abweichung.",
        },
        "head": {
            None: "Für die Kopfposition brauchen wir Schulter- und Körperkontext.",
        },
    }
    return texts[key].get(score, "")


def analyze(image_bgr):
    detector = load_detector()
    baseline, baseline_meta = load_baseline()

    face, detector_conf = crop_face(detector, image_bgr)
    if face is None:
        return {
            "ok": False,
            "message": "Kein Katzengesicht sicher erkannt. Bitte frontal und näher fotografieren.",
        }

    quality = face_quality(face, detector_conf)

    try:
        landmarks = predict_landmarks(face)
        features = extract_geometry_features(landmarks)
    except Exception:
        return {
            "ok": False,
            "message": "Das Gesicht wurde erkannt, aber die 48 Landmark-Punkte konnten nicht stabil bestimmt werden.",
            "face": face,
        }

    geometry = score_fgs_geometry(features, baseline)

    partial_total = geometry["partial_total"]
    partial_max = geometry["partial_max"]
    normalized_score = (
        partial_total / partial_max * 10.0
        if partial_max
        else 0.0
    )

    explanations = geometry["explanations"]
    scores = geometry["scores"]

    payload = build_vision_payload(
        scores=scores,
        explanations=explanations,
        normalized_score=normalized_score,
        partial_total=partial_total,
        partial_max=partial_max,
        accepted_count=1 if quality["accepted"] else 0,
        detected_count=1,
        landmark_success_count=1,
        quality_rejected_count=0 if quality["accepted"] else 1,
        landmark_failed_count=0,
        stability_index=1.0,
        baseline_cats=baseline_meta.get("healthy_cats", "?"),
        session_features=features,
    )

    return {
        "ok": True,
        "face": face,
        "landmarks": landmarks,
        "quality": quality,
        "detector_confidence": detector_conf,
        "geometry": geometry,
        "normalized_score": normalized_score,
        "payload": payload,
    }


st.markdown(
    """
<div class="hero">
  <h1>🐱 AI-Tails Live</h1>
  <p>Fotografiere die Katze auf der Leinwand. Wir erkennen das Gesicht, messen 48 Landmark-Punkte und vergleichen FGS-orientierte Merkmale mit einer gesunden Referenz.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("MAKEathon Demo · Screening-Prototyp · keine tierärztliche Diagnose")

camera = st.camera_input("📷 Katze fotografieren", help="Am besten frontal, scharf und mit gut sichtbarem Gesicht.")

with st.expander("Oder vorhandenes Bild hochladen"):
    upload = st.file_uploader(
        "Bild auswählen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

source = camera if camera is not None else upload

if source is None:
    st.info("Scanne den QR-Code, öffne die Kamera und fotografiere die Katze auf der Leinwand.")
    st.stop()

image_bgr = decode_image(source)

if image_bgr is None:
    st.error("Das Bild konnte nicht gelesen werden.")
    st.stop()

with st.spinner("AI-Tails analysiert das Katzengesicht …"):
    result = analyze(image_bgr)

if not result["ok"]:
    st.error(result["message"])
    if result.get("face") is not None:
        st.image(
            cv2.cvtColor(result["face"], cv2.COLOR_BGR2RGB),
            caption="Erkanntes Gesicht",
            use_container_width=True,
        )
    st.info("Tipp: Gesicht größer, frontal und ohne starke Bewegungsunschärfe aufnehmen.")
    st.stop()

scores = result["geometry"]["scores"]
explanations = result["geometry"]["explanations"]
normalized_score = result["normalized_score"]
title, subtitle, emoji = human_summary(normalized_score)

st.markdown(
    f"""
<div class="result-card">
  <div class="small-muted">AI-Tails Ergebnis</div>
  <h2>{emoji} {title}</h2>
  <p>{subtitle}</p>
</div>
""",
    unsafe_allow_html=True,
)

quality = result["quality"]

q1, q2, q3 = st.columns(3)
q1.metric("Gesicht erkannt", f"{result['detector_confidence'] * 100:.0f}%")
q2.metric("Landmarks", "48 / 48")
q3.metric("Bildqualität", "Gut" if quality["accepted"] else "Eingeschränkt")

if not quality["accepted"]:
    st.warning(
        "Die Analyse ist möglich, aber die Bildqualität ist eingeschränkt: "
        + ", ".join(quality["reasons"])
    )

st.subheader("Was sehen wir im Gesicht?")

labels = {
    "ears": "Ohren",
    "eyes": "Augen",
    "muzzle": "Muzzle",
    "whiskers": "Whisker-Region",
    "head": "Kopfposition",
}

cols = st.columns(2)

for index, key in enumerate(("ears", "eyes", "muzzle", "whiskers", "head")):
    with cols[index % 2]:
        score = scores[key]
        info = explanations[key]

        if score is None:
            st.markdown(f"**{labels[key]} · nicht bewertet**")
            st.caption(feature_explanation(key, score))
        else:
            dots = "●" * score + "○" * (2 - score)
            st.markdown(f"**{labels[key]} · {score}/2 · {dots}**")
            st.caption(feature_explanation(key, score))
            st.caption(f"Abweichung: {info['deviation_sigma']:.2f}σ")

st.caption(
    "0 = nahe an der gesunden Referenz · 1 = moderate Abweichung · 2 = deutliche Abweichung"
)

with st.expander("✨ 48 Landmark-Punkte auf dem Gesicht ansehen"):
    overlay = draw_landmarks(result["face"], result["landmarks"])
    st.image(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        caption="48 CatFLW Landmark-Punkte",
        use_container_width=True,
    )
    st.caption(
        "Diese Punkte bilden u. a. Augen, Ohren, Nase und Maul ab. "
        "Aus ihren relativen Abständen und Winkeln berechnen wir die geometrischen Merkmale."
    )

with st.expander("Technische Details"):
    st.write(
        f"Gesunde Referenz: {result['payload']['reference']['healthy_baseline_cats']} Katzen"
    )
    st.write(f"Schärfewert: {quality['sharpness']:.0f}")
    st.write(f"Gesichtsgröße: {quality['width']} × {quality['height']} px")
    st.json(result["payload"])

st.divider()
st.markdown(
    "**Nächster Systemschritt:** Dieser strukturierte Vision-Output wird im Gesamtsystem an den Graphwise Knowledge Graph übergeben. Dort werden die Merkmale fachlich eingeordnet; ein Agent formuliert daraus anschließend eine verständliche Erklärung."
)
