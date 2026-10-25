from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Iterable

import numpy as np


# Exact CatFLW / cat_detection 48-point topology.
IDX = {
    "chin_center": 0,
    "left_face_contour": 1,
    "muzzle_center": 2,
    "right_face_contour": 3,

    "right_eye_outer": 4,
    "right_eye0": 5,
    "right_eye1": 6,
    "right_eye2": 7,

    "left_eye_outer": 8,
    "left_eye0": 9,
    "left_eye1": 10,
    "left_eye2": 11,

    "nose_left": 12,
    "nose_right": 13,
    "nose_bridge_left": 14,
    "nose_bridge_right": 15,

    "mouth_top": 16,
    "mouth_bottom": 17,
    "chin_left0": 18,
    "chin_left1": 19,
    "chin_right0": 20,
    "chin_right1": 21,

    "right_ear0": 22,
    "right_ear1": 23,
    "right_ear2": 24,
    "right_ear3": 25,
    "right_ear4": 26,

    "left_ear0": 27,
    "left_ear1": 28,
    "left_ear2": 29,
    "left_ear3": 30,
    "left_ear4": 31,

    "nose_ring_left0": 32,
    "nose_ring_left1": 33,
    "nose_ring_right0": 34,
    "nose_ring_right1": 35,

    "right_eye_bottom": 36,
    "right_eye_inner": 37,
    "right_eye_top": 38,

    "left_eye_top": 39,
    "left_eye_inner": 40,
    "left_eye_bottom": 41,

    "nose_tip_left": 42,
    "nose_tip_right": 43,
    "nose_wing_left": 44,
    "nose_wing_right": 45,

    "mouth_corner_left": 46,
    "mouth_corner_right": 47,
}


@dataclass
class HealthyBaseline:
    mean: dict[str, float]
    std: dict[str, float]


def _pt(points: np.ndarray, name: str) -> np.ndarray:
    return points[IDX[name]]


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(den, 1e-6))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    return float(degrees(atan2(v[1], v[0])))


def _validate(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (48, 2):
        raise ValueError(f"Expected landmarks with shape (48, 2), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Landmarks contain NaN/inf values.")
    return arr


def extract_geometry_features(points: Iterable[Iterable[float]]) -> dict[str, float]:
    """
    Compute scale-normalized, interpretable CatFLW geometry features.

    All ratios are normalized by stable face dimensions so image scale does not
    directly affect the result.
    """
    p = _validate(points)

    eye_center_left = (
        _pt(p, "left_eye_outer")
        + _pt(p, "left_eye_inner")
        + _pt(p, "left_eye_top")
        + _pt(p, "left_eye_bottom")
    ) / 4.0

    eye_center_right = (
        _pt(p, "right_eye_outer")
        + _pt(p, "right_eye_inner")
        + _pt(p, "right_eye_top")
        + _pt(p, "right_eye_bottom")
    ) / 4.0

    inter_eye = _distance(eye_center_left, eye_center_right)
    face_width = _distance(
        _pt(p, "left_face_contour"),
        _pt(p, "right_face_contour"),
    )

    norm = max(inter_eye, face_width, 1e-6)

    left_eye_width = _distance(
        _pt(p, "left_eye_outer"),
        _pt(p, "left_eye_inner"),
    )
    left_eye_height = _distance(
        _pt(p, "left_eye_top"),
        _pt(p, "left_eye_bottom"),
    )

    right_eye_width = _distance(
        _pt(p, "right_eye_outer"),
        _pt(p, "right_eye_inner"),
    )
    right_eye_height = _distance(
        _pt(p, "right_eye_top"),
        _pt(p, "right_eye_bottom"),
    )

    left_eye_open = _safe_ratio(left_eye_height, left_eye_width)
    right_eye_open = _safe_ratio(right_eye_height, right_eye_width)
    eye_open = (left_eye_open + right_eye_open) / 2.0

    # Ear contour endpoints and central/tip-like point.
    # We use contour span and orientation relative to eye line, not raw pixels.
    right_ear_base = (_pt(p, "right_ear0") + _pt(p, "right_ear4")) / 2.0
    right_ear_tip = _pt(p, "right_ear2")
    left_ear_base = (_pt(p, "left_ear0") + _pt(p, "left_ear4")) / 2.0
    left_ear_tip = _pt(p, "left_ear2")

    right_ear_angle = _angle_deg(right_ear_base, right_ear_tip)
    left_ear_angle = _angle_deg(left_ear_base, left_ear_tip)

    # Convert to an "outward rotation" proxy where larger is more lateral.
    # Absolute geometry is later calibrated against healthy cats.
    ear_rotation_asym = abs(abs(left_ear_angle) - abs(right_ear_angle))
    ear_tip_separation = _safe_ratio(
        _distance(left_ear_tip, right_ear_tip),
        norm,
    )
    ear_base_separation = _safe_ratio(
        _distance(left_ear_base, right_ear_base),
        norm,
    )
    ear_spread_ratio = _safe_ratio(ear_tip_separation, ear_base_separation)

    mouth_width = _distance(
        _pt(p, "mouth_corner_left"),
        _pt(p, "mouth_corner_right"),
    )
    mouth_height = _distance(
        _pt(p, "mouth_top"),
        _pt(p, "mouth_bottom"),
    )

    nose_width = _distance(
        _pt(p, "nose_left"),
        _pt(p, "nose_right"),
    )

    muzzle_width = _distance(
        _pt(p, "nose_wing_left"),
        _pt(p, "nose_wing_right"),
    )

    muzzle_to_mouth = _safe_ratio(muzzle_width, mouth_width)
    mouth_aspect = _safe_ratio(mouth_height, mouth_width)
    nose_to_mouth = _safe_ratio(nose_width, mouth_width)

    # Lower-face compactness: vertical nose/muzzle-to-chin distance relative
    # to face width. Useful as an interpretable tension/shape proxy.
    lower_face_height = _distance(
        _pt(p, "muzzle_center"),
        _pt(p, "chin_center"),
    )
    lower_face_compactness = _safe_ratio(lower_face_height, norm)

    face_mid_x = (
        _pt(p, "nose_bridge_left")[0]
        + _pt(p, "nose_bridge_right")[0]
    ) / 2.0

    left_mouth_offset = abs(_pt(p, "mouth_corner_left")[0] - face_mid_x)
    right_mouth_offset = abs(_pt(p, "mouth_corner_right")[0] - face_mid_x)
    mouth_asymmetry = _safe_ratio(
        abs(left_mouth_offset - right_mouth_offset),
        norm,
    )

    return {
        "eye_open_ratio": float(eye_open),
        "eye_asymmetry": float(abs(left_eye_open - right_eye_open)),
        "ear_spread_ratio": float(ear_spread_ratio),
        "ear_rotation_asymmetry": float(ear_rotation_asym),
        "muzzle_to_mouth_ratio": float(muzzle_to_mouth),
        "mouth_aspect_ratio": float(mouth_aspect),
        "nose_to_mouth_ratio": float(nose_to_mouth),
        "lower_face_compactness": float(lower_face_compactness),
        "mouth_asymmetry": float(mouth_asymmetry),
    }


def fit_healthy_baseline(feature_rows: list[dict[str, float]]) -> HealthyBaseline:
    if len(feature_rows) < 3:
        raise ValueError("Need at least 3 healthy reference samples.")

    keys = sorted(feature_rows[0].keys())
    matrix = np.asarray(
        [[row[key] for key in keys] for row in feature_rows],
        dtype=np.float32,
    )

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    std = np.maximum(std, 1e-4)

    return HealthyBaseline(
        mean={k: float(v) for k, v in zip(keys, mean)},
        std={k: float(v) for k, v in zip(keys, std)},
    )


def zscore(value: float, mean: float, std: float) -> float:
    return float((value - mean) / max(std, 1e-4))


def _severity_from_abs_z(abs_z: float) -> int:
    # Conservative initial thresholds. These must later be calibrated on the
    # project healthy/sick data. They are intentionally transparent.
    if abs_z < 1.0:
        return 0
    if abs_z < 2.0:
        return 1
    return 2


def score_fgs_geometry(
    features: dict[str, float],
    baseline: HealthyBaseline,
) -> dict:
    """
    Convert deviations from the healthy baseline into explainable FGS-oriented
    geometric scores.

    Head position is intentionally NOT inferred here because true FGS head
    position requires the shoulder line/body context, not only a face crop.
    Whiskers are approximated from muzzle geometry and should be treated as a
    proxy until dedicated whisker landmarks/image features are available.
    """

    z = {
        key: zscore(
            features[key],
            baseline.mean[key],
            baseline.std[key],
        )
        for key in features
        if key in baseline.mean
    }

    # Orbital tightening: reduced eye opening relative to healthy baseline.
    eyes_z = max(0.0, -z["eye_open_ratio"])
    eyes_score = _severity_from_abs_z(eyes_z)

    # Ear position: use spread change + abnormal asymmetry.
    ear_z = max(
        abs(z["ear_spread_ratio"]),
        abs(z["ear_rotation_asymmetry"]),
    )
    ears_score = _severity_from_abs_z(ear_z)

    # Muzzle tension: strongest lower-face geometric deviation.
    muzzle_z = max(
        abs(z["muzzle_to_mouth_ratio"]),
        abs(z["mouth_aspect_ratio"]),
        abs(z["lower_face_compactness"]),
    )
    muzzle_score = _severity_from_abs_z(muzzle_z)

    # Whisker change proxy: CatFLW has no individual whisker-line landmarks.
    # Use muzzle/nose shape + asymmetry as an explainable geometric proxy.
    whisker_z = max(
        abs(z["nose_to_mouth_ratio"]),
        abs(z["mouth_asymmetry"]),
    )
    whiskers_score = _severity_from_abs_z(whisker_z)

    scores = {
        "ears": ears_score,
        "eyes": eyes_score,
        "muzzle": muzzle_score,
        "whiskers": whiskers_score,
        "head": None,
    }

    explanations = {
        "ears": {
            "score": ears_score,
            "deviation_sigma": round(float(ear_z), 2),
            "reason": "Ohrstellung weicht geometrisch von der gesunden Referenz ab.",
        },
        "eyes": {
            "score": eyes_score,
            "deviation_sigma": round(float(eyes_z), 2),
            "reason": "Augenöffnung ist gegenüber der gesunden Referenz reduziert.",
        },
        "muzzle": {
            "score": muzzle_score,
            "deviation_sigma": round(float(muzzle_z), 2),
            "reason": "Muzzle-/Maulgeometrie weicht von der gesunden Referenz ab.",
        },
        "whiskers": {
            "score": whiskers_score,
            "deviation_sigma": round(float(whisker_z), 2),
            "reason": "Whisker-/Muzzle-Region zeigt eine geometrische Abweichung.",
        },
        "head": {
            "score": None,
            "deviation_sigma": None,
            "reason": "Kopfposition benötigt Schulter-/Körperkontext und wird aus einem Face-Crop nicht seriös geschätzt.",
        },
    }

    known_scores = [v for v in scores.values() if v is not None]

    return {
        "scores": scores,
        "partial_total": int(sum(known_scores)),
        "partial_max": int(2 * len(known_scores)),
        "z_scores": z,
        "explanations": explanations,
    }
