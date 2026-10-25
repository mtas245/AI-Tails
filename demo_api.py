from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import base64
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from fgs_geometry import HealthyBaseline, extract_geometry_features, score_fgs_geometry
from landmark_detector import predict_landmarks, draw_landmarks


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
YOLO_MODEL = ROOT / "models" / "cat_face_yolov8n.pt"
BASELINE_FILE = ROOT / "models" / "healthy_geometry_baseline.json"

app = FastAPI(title="AI-Tails Demo API", version="1.0")


@lru_cache(maxsize=1)
def load_detector():
    if not YOLO_MODEL.exists():
        raise RuntimeError("Missing models/cat_face_yolov8n.pt")
    return YOLO(str(YOLO_MODEL))


@lru_cache(maxsize=1)
def load_baseline():
    if not BASELINE_FILE.exists():
        raise RuntimeError("Missing models/healthy_geometry_baseline.json")
    raw = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    return HealthyBaseline(
        mean={k: float(v) for k, v in raw["mean"].items()},
        std={k: float(v) for k, v in raw["std"].items()},
    ), raw


def crop_face(image_bgr: np.ndarray):
    results = load_detector().predict(
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
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * 0.10), int(bh * 0.10)

    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    return crop, float(confs[idx])


def quality(face_bgr: np.ndarray, detector_confidence: float):
    h, w = face_bgr.shape[:2]
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if detector_confidence < 0.50:
        reasons.append("Gesicht nur unsicher erkannt")
    if min(h, w) < 80:
        reasons.append("Gesicht zu klein")
    if sharpness < 30:
        reasons.append("Bild zu unscharf")
    return {
        "accepted": not reasons,
        "sharpness": round(sharpness, 1),
        "width": int(w),
        "height": int(h),
        "reasons": reasons,
    }


def status_from_score(score: float):
    if score < 2.5:
        return {
            "level": "healthy",
            "eyebrow": "AI-Tails Screening",
            "title": "Healthy-like",
            "summary": "Die aktuell messbaren Gesichtssignale liegen überwiegend nahe an der gesunden Referenz.",
        }
    if score < 5.0:
        return {
            "level": "slight",
            "eyebrow": "AI-Tails Screening",
            "title": "Slight indicators",
            "summary": "Es sind leichte Abweichungen erkennbar. Einzelne Signale sollten im Verlauf weiter beobachtet werden.",
        }
    if score < 7.5:
        return {
            "level": "monitor",
            "eyebrow": "AI-Tails Screening",
            "title": "Monitoring",
            "summary": "Mehrere Gesichtssignale weichen von der gesunden Referenz ab. Eine erneute Beobachtung ist sinnvoll.",
        }
    return {
        "level": "vet",
        "eyebrow": "AI-Tails Screening",
        "title": "Vet check",
        "summary": "Mehrere deutliche Abweichungen wurden erkannt. Eine tierärztliche Beurteilung sollte in Betracht gezogen werden.",
    }


def feature_text(key: str, score: int | None):
    labels = {
        "ears": "Ohren",
        "eyes": "Augen",
        "muzzle": "Schnauze",
        "whiskers": "Whisker-/Muzzle-Region",
        "head": "Kopfposition",
    }
    if score is None:
        text = "Benötigt Schulter- und Körperkontext und wird in dieser Demo nicht bewertet."
    elif score == 0:
        text = "Liegt nahe an der gesunden Referenz."
    elif score == 1:
        text = "Zeigt eine moderate Abweichung von der gesunden Referenz."
    else:
        text = "Zeigt eine deutliche Abweichung von der gesunden Referenz."
    return labels[key], text


def encode_jpeg(image_bgr: np.ndarray):
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Bild konnte nicht gelesen werden.")

    face, detector_conf = crop_face(image)
    if face is None:
        raise HTTPException(
            status_code=422,
            detail="Kein Katzengesicht sicher erkannt. Bitte näher und möglichst frontal fotografieren.",
        )

    q = quality(face, detector_conf)

    try:
        landmarks = predict_landmarks(face)
        features = extract_geometry_features(landmarks)
        baseline, meta = load_baseline()
        geometry = score_fgs_geometry(features, baseline)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Landmark-Analyse fehlgeschlagen: {exc}") from exc

    partial_total = geometry["partial_total"]
    partial_max = geometry["partial_max"]
    normalized = (partial_total / partial_max * 10.0) if partial_max else 0.0

    overlay = draw_landmarks(face, landmarks)

    feature_cards = []
    for key in ("ears", "eyes", "muzzle", "whiskers", "head"):
        score = geometry["scores"][key]
        label, text = feature_text(key, score)
        feature_cards.append({
            "key": key,
            "label": label,
            "score": score,
            "max": 2 if score is not None else None,
            "deviation_sigma": geometry["explanations"][key]["deviation_sigma"],
            "text": text,
        })

    report = status_from_score(normalized)
    report["note"] = (
        "AI-Tails erkennt Abweichungen in messbaren Gesichtsmerkmalen. "
        "Ein einzelnes Foto ist keine tierärztliche Diagnose und sollte im Kontext weiterer Signale betrachtet werden."
    )

    return {
        "ok": True,
        "status": report,
        "normalized_score_0_10": round(normalized, 2),
        "partial_score": partial_total,
        "partial_max": partial_max,
        "detector_confidence": round(detector_conf, 4),
        "quality": q,
        "healthy_reference_cats": int(meta.get("healthy_cats", 0)),
        "features": feature_cards,
        "landmark_count": 48,
        "face_image": encode_jpeg(face),
        "landmark_image": encode_jpeg(overlay),
        "limitations": [
            "Screening-Prototyp, keine Diagnose.",
            "Kopfposition benötigt Schulter-/Körperkontext.",
            "Whisker werden aktuell über Muzzle-Geometrie angenähert.",
        ],
    }


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
