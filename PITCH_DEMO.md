# AI-Tails Pitch Demo Runbook

## Audience flow

```text
QR scannen
    -> mobile Web-App
    -> Katze fotografieren
    -> Analyse
    -> Screening-Status
    -> Gesichtssignale
    -> Landmark-Ansicht
```

## Terminal 1 — demo app

```powershell
cd C:\Users\Kirta\AI-Tails
.\.venv\Scripts\Activate.ps1
python -m uvicorn demo_api:app --host 0.0.0.0 --port 8000
```

Local URL:

```text
http://localhost:8000
```

Keep this terminal open.

## Terminal 2 — public HTTPS tunnel

```powershell
cd C:\Users\Kirta\AI-Tails
.\.venv\Scripts\Activate.ps1
& "$env:USERPROFILE\cloudflared.exe" tunnel --url http://localhost:8000
```

Copy the newly generated `https://...trycloudflare.com` URL.

Keep this terminal open.

## Terminal 3 — QR code

```powershell
cd C:\Users\Kirta\AI-Tails
.\.venv\Scripts\Activate.ps1
python make_qr.py https://YOUR-NEW-URL.trycloudflare.com
```

Generated file:

```text
pitch_qr.png
```

## Before presenting

- run `git pull`
- run `python -m pip install -r requirements.txt`
- run `python validate_mvp.py`
- start Terminal 1
- test `http://localhost:8000`
- perform one full analysis to warm the models
- start Terminal 2
- test the public URL from a phone
- create the QR in Terminal 3
- scan the QR once before presenting
- leave Terminal 1 and Terminal 2 running

## Pitch safety / fallback

Keep one known-good cat image locally.

If camera access fails:
- use image upload.

If the tunnel fails:
- demonstrate locally at `http://localhost:8000`.

If image quality is insufficient:
- the app should return `Analyse unsicher` rather than a health category.

Do not present the four screening states as a veterinary diagnosis.

## Current demo states

- Healthy-like
- Slight indicators
- Monitoring
- Vet check
- Analyse unsicher (quality fallback)

## Known limitation to state if asked

Different photos of the same cat can still produce different deviations because camera angle, distance, zoom, pose and image quality affect facial geometry. Multi-frame stabilization is planned as follow-up work.
