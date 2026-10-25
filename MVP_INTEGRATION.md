# AI-Tails MVP Integration Contract

## Purpose

The Vision MVP is responsible for **measurement**, not final medical interpretation.

The intended system boundary is:

```text
Images / videos
  -> Cat face detection
  -> 48 CatFLW facial landmarks
  -> FGS-oriented geometric measurements
  -> Healthy-baseline comparison
  -> JSON payload
  -> Graphwise Knowledge Graph
  -> Agent
  -> Owner / vet-facing explanation
```

Graphwise and the downstream agent should treat the Vision payload as evidence, not as a diagnosis.

## Stable JSON contract

The Streamlit demo exports payloads with:

```json
{
  "schema_version": "ai-tails.vision.v1",
  "source": "ai-tails-vision-mvp",
  "assessment_type": "fgs_oriented_geometry_screening",
  "fgs_features": {
    "ears": {
      "score": 0,
      "deviation_sigma": 0.4,
      "measurement_note": "..."
    },
    "eyes": {
      "score": 1,
      "deviation_sigma": 1.3,
      "measurement_note": "..."
    },
    "muzzle": {
      "score": 0,
      "deviation_sigma": 0.7,
      "measurement_note": "..."
    },
    "whiskers": {
      "score": 2,
      "deviation_sigma": 2.4,
      "measurement_note": "..."
    },
    "head": {
      "score": null,
      "deviation_sigma": null,
      "measurement_note": "..."
    }
  },
  "geometry": {
    "partial_score": 3,
    "partial_max": 8,
    "normalized_score_0_10": 3.75
  },
  "quality": {
    "detected_face_observations": 14,
    "accepted_for_geometry": 10,
    "landmark_success_count": 13,
    "quality_rejected_count": 3,
    "landmark_failed_count": 1,
    "session_stability_0_1": 0.82
  },
  "reference": {
    "healthy_baseline_cats": 4,
    "baseline_type": "project_specific_healthy_geometry"
  }
}
```

## Interpretation boundary

The Vision component may say:

- an ear geometry is 2.1 standard deviations from the healthy baseline;
- an eye-opening ratio is reduced;
- 10 of 14 detected observations passed quality filters;
- a feature received a transparent project score of 0, 1, or 2.

The Vision component must **not** claim:

- a clinical diagnosis;
- a probability that the cat is ill;
- a specific disease;
- that the project-specific thresholds are clinically validated.

Those interpretations belong in Graphwise / the agent layer, where additional knowledge and rules can be applied.

## Integration rule

Downstream systems should depend on `schema_version`, not on Streamlit UI text.

If the payload format changes incompatibly, increment the schema version.
