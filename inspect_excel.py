import pandas as pd
from pathlib import Path
import re

excel_path = Path("data/labels/all_cats.xlsx")
df = pd.read_excel(excel_path, sheet_name="Data")

pain_col = "Pain level according to the Glasgow scale"
cat_col = "CatID/Name"
photo_col = "Number of photos and videos"

def extract_score(value):
    if pd.isna(value):
        return None

    match = re.match(r"\s*(\d+)", str(value))
    if match:
        return int(match.group(1))

    return None

df["pain_score"] = df[pain_col].apply(extract_score)

df["pain_binary"] = df["pain_score"].apply(
    lambda x: "unauffaellig" if x is not None and x <= 3 else "auffaellig"
)

print("=== DATENÜBERSICHT ===")

print("\nZeilen insgesamt:")
print(len(df))

print("\nEinzigartige Katzen insgesamt:")
print(df[cat_col].nunique())

print("\nZeilen pro Klasse:")
print(df["pain_binary"].value_counts())

print("\nEinzigartige Katzen pro Klasse:")
print(
    df.groupby("pain_binary")[cat_col]
      .nunique()
)

print("\nFotos/Videos pro Klasse:")
print(
    df.groupby("pain_binary")[photo_col]
      .sum()
)

print("\nDurchschnittliche Anzahl Bilder/Videos pro Datensatz:")
print(
    df.groupby("pain_binary")[photo_col]
      .mean()
)