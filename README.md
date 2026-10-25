# AI-Tails Vision MVP

AI-Tails is a MAKEathon proof of concept for explainable cat facial screening.

## MVP scope

The Vision component:

1. accepts one or more images and videos of the same cat;
2. detects the cat face with YOLO;
3. predicts 48 CatFLW facial landmarks;
4. computes FGS-oriented geometric features;
5. compares those measurements with a local healthy reference baseline;
6. aggregates multiple observations robustly using the median;
7. exports a versioned JSON payload for Graphwise / the downstream agent.

The Vision component is a **screening / measurement system**, not a veterinary diagnosis.

## Architecture

```text
Images / videos
    -> YOLO cat-face detector
    -> 48 CatFLW landmarks
    -> geometric features
    -> healthy-baseline comparison
    -> quality filtering + multi-frame aggregation
    -> ai-tails.vision.v1 JSON
    -> Graphwise Knowledge Graph
    -> Agent
    -> Owner / vet explanation
```

## Required local assets

These files are intentionally not committed to the public repository:

```text
models/cat_face_yolov8n.pt
models/healthy_geometry_baseline.json
```

Optional legacy model:

```text
models/health_classifier.pt
```

The legacy 3-class classifier is not part of the MVP main decision and the app runs without it.

## Setup

```powershell
python -m pip install -r requirements.txt
```

If the healthy baseline has not been created yet:

```powershell
python build_healthy_baseline.py
```

Run the demo:

```powershell
python -m streamlit run app.py
```

## Pre-pitch check

```powershell
python validate_mvp.py
```

The check verifies the required local files, healthy-baseline schema, importable Python modules, and integration contract version.

## Integration

See [MVP_INTEGRATION.md](MVP_INTEGRATION.md).

The stable downstream interface is:

```text
schema_version = ai-tails.vision.v1
```

Downstream Graphwise / agent components should integrate against the JSON schema, not Streamlit text.

## Current limitations

- Healthy reference currently contains only a small number of cats.
- Head position is not automatically scored because true FGS head position needs shoulder/body context.
- Whisker change is approximated from whisker-pad / muzzle geometry.
- Project thresholds are transparent engineering thresholds, not clinically validated FGS thresholds.
- Results may be affected by breed morphology, head pose, occlusion, blur, lighting, and camera angle.
