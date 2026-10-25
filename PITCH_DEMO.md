# AI-Tails Pitch Demo Runbook

## Goal

Audience flow:

1. QR code scannen
2. Web-App auf dem Smartphone öffnen
3. Katze auf der Leinwand fotografieren
4. AI-Tails analysiert das Gesicht
5. Ergebnis + FGS-orientierte Merkmale ansehen
6. Optional: 48 Landmark-Punkte direkt auf dem Gesicht öffnen

## Pitch app

Run locally:

```powershell
git pull
python -m pip install -r requirements.txt
python -m streamlit run pitch_app.py
```

Open locally first:

```text
http://localhost:8501
```

Take one test photo before the pitch. This warms up YOLO, the landmark model and model caches.

## Public URL option A: quick demo tunnel

Use a Cloudflare Quick Tunnel:

```powershell
cloudflared tunnel --url http://localhost:8501
```

Cloudflare prints a public HTTPS URL such as:

```text
https://example-random.trycloudflare.com
```

Generate the QR code:

```powershell
python make_qr.py https://example-random.trycloudflare.com
```

The resulting `pitch_qr.png` can be shown in the pitch.

IMPORTANT: Quick Tunnel URLs change when restarted. Create the QR only after the final tunnel is running.

## Public URL option B: stable pitch URL

For a fixed QR code prepared in advance, use a named Cloudflare Tunnel with a stable hostname/domain.

Example architecture:

```text
https://demo.your-domain.ch
          |
    Cloudflare Tunnel
          |
http://localhost:8501
          |
    pitch_app.py
```

## Pre-pitch checklist

- Laptop connected to power
- Disable sleep mode for the pitch
- Confirm internet connection
- Run `python validate_mvp.py`
- Start `pitch_app.py`
- Open the app locally
- Perform one full test capture
- Start the public tunnel
- Test the public URL from a phone using mobile data
- Test camera permission
- Generate/test QR code
- Keep the browser tab open
- Do not restart Streamlit or the tunnel after generating a Quick-Tunnel QR

## Reliability strategy

Do not ask the full room to submit analyses simultaneously on one laptop.

Recommended pitch flow:

- Everyone may scan the QR and open the app.
- Ask 1–3 people to perform the live capture at the same time.
- Keep a second phone ready for the presenter.
- Keep one known-good cat image saved locally as an upload fallback.

This preserves the genuine live demo while avoiding CPU overload from many simultaneous model inferences.

## Fallback ladder

If phone camera permission fails:
- use the app's image-upload fallback.

If the public tunnel fails:
- presenter continues on the local app at `localhost:8501`.

If a photo contains no usable cat face:
- retake closer and more frontal.
- or upload the known-good fallback image.

If Graphwise / Agent integration is not available:
- the Vision MVP still displays the measurements, feature scores and Landmark viewer.

## What the audience should see

The main page intentionally avoids the legacy classifier.

Primary result:

- face detection quality
- 48/48 landmarks
- image quality
- ears / eyes / muzzle / whisker-region scores
- owner-readable short explanation
- optional landmark overlay
- technical JSON hidden in an expander

This makes the demonstration understandable without exposing internal development metrics.
