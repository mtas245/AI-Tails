# AI-Tails Vision MVP

AI-Tails is a MAKEathon proof of concept for explainable cat facial screening.

## What the submission contains

The current submission includes two interfaces:

- **Primary demo web app:** custom mobile frontend + FastAPI backend
- **Technical MVP app:** Streamlit interface for development and inspection

The Vision pipeline itself is shared by both interfaces.

## Vision pipeline

```text
Photo / video
    -> YOLO cat-face detector
    -> 48 CatFLW landmarks
    -> geometric facial features
    -> healthy-reference comparison
    -> conservative multi-feature screening logic
    -> result + landmark visualization
```

The system is a **screening / measurement prototype**, not a veterinary diagnosis.

## Current screening states

The demo reports one of four user-facing screening states:

- Healthy-like
- Slight indicators
- Monitoring
- Vet check

A low-quality image can instead return **Analyse unsicher**.

These states are engineering rules for the MAKEathon prototype and are not clinically validated diagnostic categories.

## Required local assets

These files are intentionally excluded from the public repository:

```text
models/cat_face_yolov8n.pt
models/healthy_geometry_baseline.json
```

Optional legacy model:

```text
models/health_classifier.pt
```

The legacy classifier is not used for the primary demo decision.

## Setup

Recommended on Windows PowerShell:

```powershell
cd C:\Users\Kirta\AI-Tails
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Validate the installation

```powershell
python validate_mvp.py
```

## Run the primary demo

```powershell
python -m uvicorn demo_api:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000
```

The mobile demo provides:

- live camera
- 1× / 2× / 3× zoom controls
- image upload fallback
- face detection quality
- 48 facial landmarks
- FGS-oriented feature scores
- four-level screening status
- large / fullscreen landmark viewer

## Optional public demo via Cloudflare Tunnel

```powershell
& "$env:USERPROFILE\cloudflared.exe" tunnel --url http://localhost:8000
```

Cloudflare prints a temporary public HTTPS URL.

Generate a QR code for that URL:

```powershell
python make_qr.py https://YOUR-URL.trycloudflare.com
```

Quick Tunnel URLs change whenever the tunnel is restarted.

## Technical Streamlit MVP

The older technical interface remains available:

```powershell
python -m streamlit run app.py
```

It exposes additional development information and the versioned downstream JSON contract.

## Integration

See [MVP_INTEGRATION.md](MVP_INTEGRATION.md).

Stable downstream schema:

```text
ai-tails.vision.v1
```

## Important limitations

- Healthy reference currently contains only a small number of individual cats.
- Results can vary with camera angle, zoom, pose, breed morphology, blur, occlusion and lighting.
- Current live demo evaluates one captured image; multi-frame stabilization is future work.
- Head position is not automatically scored because correct FGS head position needs shoulder/body context.
- Whisker change is approximated from muzzle-region geometry.
- Thresholds and screening rules are transparent engineering rules, not clinically validated FGS thresholds.
- A result must not be interpreted as a diagnosis of pain or disease.

## Main files

```text
demo_api.py              FastAPI demo backend
frontend/                Mobile demo UI
fgs_geometry.py          Interpretable geometry + scoring
landmark_detector.py     48-point CatFLW landmark inference
vision_contract.py       Stable downstream JSON contract
app.py                   Technical Streamlit MVP
validate_mvp.py          Local pre-demo validation
make_qr.py               QR-code helper
```
