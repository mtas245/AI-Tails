from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "ai-tails.vision.v1"


def build_vision_payload(
    *,
    scores: dict[str, int | None],
    explanations: dict[str, dict[str, Any]],
    normalized_score: float,
    partial_total: int,
    partial_max: int,
    accepted_count: int,
    detected_count: int,
    landmark_success_count: int,
    quality_rejected_count: int,
    landmark_failed_count: int,
    stability_index: float,
    baseline_cats: int | str,
    session_features: dict[str, float],
) -> dict[str, Any]:
    """
    Stable, JSON-serializable contract between the vision MVP and the
    Graphwise / agent layer.

    This payload intentionally contains measurements and quality metadata,
    not a medical diagnosis.
    """

    fgs_features = {}

    for key in ("ears", "eyes", "muzzle", "whiskers", "head"):
        info = explanations[key]
        fgs_features[key] = {
            "score": scores[key],
            "deviation_sigma": info.get("deviation_sigma"),
            "measurement_note": info.get("reason"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ai-tails-vision-mvp",
        "assessment_type": "fgs_oriented_geometry_screening",
        "fgs_features": fgs_features,
        "geometry": {
            "partial_score": partial_total,
            "partial_max": partial_max,
            "normalized_score_0_10": round(float(normalized_score), 3),
            "raw_session_features": {
                key: round(float(value), 6)
                for key, value in session_features.items()
            },
        },
        "quality": {
            "detected_face_observations": int(detected_count),
            "accepted_for_geometry": int(accepted_count),
            "landmark_success_count": int(landmark_success_count),
            "quality_rejected_count": int(quality_rejected_count),
            "landmark_failed_count": int(landmark_failed_count),
            "session_stability_0_1": round(float(stability_index), 4),
        },
        "reference": {
            "healthy_baseline_cats": baseline_cats,
            "baseline_type": "project_specific_healthy_geometry",
        },
        "limitations": [
            "Screening output only; not a veterinary diagnosis.",
            "Head position is not scored until shoulder/body context is available.",
            "Whiskers are currently approximated from whisker-pad/muzzle geometry.",
            "Thresholds and the healthy baseline are not clinically validated.",
        ],
    }
