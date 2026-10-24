from pathlib import Path
import pandas as pd
import re
import unicodedata

raw_dir = Path("data/raw")
excel_path = Path("data/labels/all_cats.xlsx")

df = pd.read_excel(excel_path, sheet_name="Data")

cat_col = "CatID/Name"

def normalize_name(name):
    name = str(name).strip().lower()

    # Umlaute / Sonderzeichen normalisieren
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Datumsangaben entfernen
    name = re.sub(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", "", name)

    # Klammer-Inhalte entfernen
    name = re.sub(r"\([^)]*\)", "", name)

    # häufige Zusätze entfernen
    extras = [
        "kitten",
        "chaton",
        "ragdoll",
        "white british lh",
    ]

    for extra in extras:
        name = name.replace(extra, "")

    # abschließende Nummern entfernen, z.B. "pepper 1"
    name = re.sub(r"\s+\d+$", "", name)

    # Mehrfach-Leerzeichen bereinigen
    name = re.sub(r"\s+", " ", name).strip()

    return name


excel_names_raw = [x for x in df[cat_col].dropna()]
excel_map = {
    normalize_name(x): str(x)
    for x in excel_names_raw
}

folder_names = [
    folder.name
    for folder in raw_dir.iterdir()
    if folder.is_dir()
]

matches = []
no_matches = []

for folder_name in folder_names:
    norm = normalize_name(folder_name)

    if norm in excel_map:
        matches.append((folder_name, excel_map[norm]))
    else:
        no_matches.append(folder_name)

print("=== MATCHING CHECK ===")

print("\nExcel-Katzen:")
print(len(excel_names_raw))

print("\nKatzenordner:")
print(len(folder_names))

print("\nAutomatische Matches:")
print(len(matches))

print("\nKeine Matches:")
print(len(no_matches))

print("\n--- NICHT GEMATCHTE ORDNER ---")
for name in no_matches:
    print(name)