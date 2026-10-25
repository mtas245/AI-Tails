from pathlib import Path
from collections import Counter

import pandas as pd


ROOT = Path(__file__).resolve().parent
NEW_DATA_DIR = ROOT / "data" / "new_data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
REPORT_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".rtf"}


def main():
    if not NEW_DATA_DIR.exists():
        print("Ordner fehlt:")
        print(NEW_DATA_DIR)
        print()
        print("Bitte erstelle ihn und kopiere die neuen Daten dort hinein.")
        return

    files = [p for p in NEW_DATA_DIR.rglob("*") if p.is_file()]
    folders = [p for p in NEW_DATA_DIR.rglob("*") if p.is_dir()]

    extensions = Counter(p.suffix.lower() for p in files)

    images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTENSIONS]
    excels = [p for p in files if p.suffix.lower() in EXCEL_EXTENSIONS]
    reports = [p for p in files if p.suffix.lower() in REPORT_EXTENSIONS]

    print("=== NEW DATA INVENTORY ===")
    print("Base folder:", NEW_DATA_DIR)
    print("Folders:", len(folders))
    print("Files:", len(files))
    print("Images:", len(images))
    print("Videos:", len(videos))
    print("Excel files:", len(excels))
    print("Report files:", len(reports))

    print("\n=== FILE TYPES ===")
    for ext, count in sorted(extensions.items()):
        print(f"{ext or '[no extension]'}: {count}")

    print("\n=== TOP-LEVEL FOLDERS ===")
    top_level = [p for p in NEW_DATA_DIR.iterdir() if p.is_dir()]
    if top_level:
        for folder in sorted(top_level, key=lambda p: p.name.lower()):
            count = sum(1 for p in folder.rglob("*") if p.is_file())
            print(f"{folder.name}: {count} files")
    else:
        print("No top-level subfolders found.")

    print("\n=== EXCEL STRUCTURE ===")
    if not excels:
        print("No Excel files found.")
    else:
        for path in sorted(excels, key=lambda p: str(p).lower()):
            print(f"\nFILE: {path.relative_to(NEW_DATA_DIR)}")
            try:
                workbook = pd.ExcelFile(path)
                print("Sheets:", ", ".join(workbook.sheet_names))

                for sheet in workbook.sheet_names:
                    try:
                        df = pd.read_excel(path, sheet_name=sheet, nrows=0)
                        columns = [str(c) for c in df.columns]
                        print(f"  [{sheet}] columns ({len(columns)}):")
                        for column in columns:
                            print(f"    - {column}")
                    except Exception as exc:
                        print(f"  [{sheet}] could not read columns: {exc}")

            except Exception as exc:
                print("Could not open workbook:", exc)

    print("\n=== REPORT FILES ===")
    if reports:
        for path in sorted(reports, key=lambda p: str(p).lower()):
            print(path.relative_to(NEW_DATA_DIR))
    else:
        print("No report files found.")

    print("\n=== MEDIA BY TOP-LEVEL FOLDER ===")
    for folder in sorted(top_level, key=lambda p: p.name.lower()):
        folder_files = [p for p in folder.rglob("*") if p.is_file()]
        image_count = sum(
            1 for p in folder_files if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        video_count = sum(
            1 for p in folder_files if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        report_count = sum(
            1 for p in folder_files if p.suffix.lower() in REPORT_EXTENSIONS
        )
        excel_count = sum(
            1 for p in folder_files if p.suffix.lower() in EXCEL_EXTENSIONS
        )

        print(
            f"{folder.name}: "
            f"images={image_count}, "
            f"videos={video_count}, "
            f"reports={report_count}, "
            f"excel={excel_count}"
        )

    print("\nDone.")
    print(
        "This script only inventories local files. "
        "It does not upload the medical data or media."
    )


if __name__ == "__main__":
    main()
