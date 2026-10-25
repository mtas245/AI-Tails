from __future__ import annotations

from pathlib import Path
import csv

import pandas as pd
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
FACES_DIR = ROOT / "data" / "faces"
LABEL_FILE = ROOT / "data" / "fgs_labels.csv"

ACTION_UNITS = [
    "ears",
    "eyes",
    "muzzle",
    "whiskers",
    "head",
]

DISPLAY_NAMES = {
    "ears": "Ohrenposition",
    "eyes": "Augenverengung",
    "muzzle": "Muzzle-/Maulspannung",
    "whiskers": "Schnurrhaare",
    "head": "Kopfposition",
}

HELP_TEXT = {
    "ears": "0 = nach vorne, 1 = leicht auseinander, 2 = nach außen rotiert",
    "eyes": "0 = offen, 1 = teilweise geschlossen, 2 = zusammengekniffen",
    "muzzle": "0 = entspannt/rund, 1 = leichte Spannung, 2 = angespannt/elliptisch",
    "whiskers": "0 = locker/gebogen, 1 = gerader/näher zusammen, 2 = gerade nach vorne",
    "head": "0 = über Schulterlinie, 1 = auf Schulterlinie, 2 = darunter/gesenkt",
}


def load_images():
    if not FACES_DIR.exists():
        return []

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        [
            path
            for path in FACES_DIR.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ],
        key=lambda p: str(p).lower(),
    )


def load_labels():
    if not LABEL_FILE.exists():
        return pd.DataFrame(
            columns=[
                "image_path",
                "cat_id",
                *ACTION_UNITS,
                "fgs_total",
                "notes",
            ]
        )

    return pd.read_csv(LABEL_FILE)


def save_label(row):
    LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "cat_id",
        *ACTION_UNITS,
        "fgs_total",
        "notes",
    ]

    exists = LABEL_FILE.exists()

    with LABEL_FILE.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    st.set_page_config(
        page_title="AI-Tails FGS Labeler",
        page_icon="🐱",
        layout="wide",
    )

    st.title("🐱 AI-Tails – FGS Labeling Tool")

    st.info(
        "Bewerte jedes Katzengesicht nach der Feline Grimace Scale. "
        "Jede Action Unit erhält 0, 1 oder 2 Punkte."
    )

    images = load_images()
    labels = load_labels()

    if not images:
        st.error(
            "Keine Face-Crops gefunden. Erwarteter Ordner: "
            f"{FACES_DIR}"
        )
        st.stop()

    labeled_paths = set(labels["image_path"].astype(str).tolist()) if not labels.empty else set()

    remaining = [
        image
        for image in images
        if str(image.relative_to(ROOT)) not in labeled_paths
    ]

    total = len(images)
    done = total - len(remaining)

    st.progress(done / total if total else 0.0)
    st.caption(f"Gelabelt: {done} / {total}")

    if not remaining:
        st.success("Alle verfügbaren Bilder wurden gelabelt.")
        st.write(f"Labels gespeichert unter: {LABEL_FILE}")
        return

    image_path = remaining[0]
    relative_path = str(image_path.relative_to(ROOT))
    cat_id = image_path.parent.name

    left, right = st.columns([1.2, 1])

    with left:
        image = Image.open(image_path).convert("RGB")
        st.image(
            image,
            caption=relative_path,
            use_container_width=True,
        )

    with right:
        st.subheader("FGS Action Units")

        values = {}

        for unit in ACTION_UNITS:
            values[unit] = st.radio(
                DISPLAY_NAMES[unit],
                options=[0, 1, 2],
                horizontal=True,
                key=f"{relative_path}_{unit}",
                help=HELP_TEXT[unit],
            )

        total_score = sum(values.values())

        st.metric("FGS-Gesamtscore", f"{total_score}/10")

        if total_score == 0:
            st.success("Kein Schmerz erkennbar")
        elif total_score <= 3:
            st.warning("Kein oder leichter Schmerz")
        elif total_score <= 8:
            st.error("Schmerz wahrscheinlich")
        else:
            st.error("Starker Schmerz wahrscheinlich")

        notes = st.text_area(
            "Notiz (optional)",
            placeholder="z. B. schlechte Bildqualität, Kopf nicht vollständig sichtbar ...",
        )

        if st.button("Label speichern & nächstes Bild", type="primary", use_container_width=True):
            row = {
                "image_path": relative_path,
                "cat_id": cat_id,
                "ears": values["ears"],
                "eyes": values["eyes"],
                "muzzle": values["muzzle"],
                "whiskers": values["whiskers"],
                "head": values["head"],
                "fgs_total": total_score,
                "notes": notes.strip(),
            }

            save_label(row)
            st.rerun()

    st.divider()

    with st.expander("FGS-Kurzreferenz"):
        st.markdown(
            """
**Ohrenposition**
- 0: nach vorne
- 1: leicht auseinander
- 2: nach außen rotiert

**Augenverengung**
- 0: Augen offen
- 1: teilweise geschlossen
- 2: zusammengekniffen

**Muzzle-/Maulspannung**
- 0: entspannt, rund
- 1: leichte Spannung
- 2: angespannt, elliptisch

**Schnurrhaare**
- 0: locker und gebogen
- 1: gerader / näher zusammen
- 2: gerade und nach vorne gerichtet

**Kopfposition**
- 0: über Schulterlinie
- 1: auf Schulterlinie
- 2: darunter oder gesenkt
"""
        )


if __name__ == "__main__":
    main()
