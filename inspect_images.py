from pathlib import Path
from collections import Counter

raw_dir = Path("data/raw")

image_extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

video_extensions = {
    ".mp4", ".mov", ".avi", ".mkv", ".m4v"
}

folders = [
    p for p in raw_dir.iterdir()
    if p.is_dir()
]

all_files = [
    p for p in raw_dir.rglob("*")
    if p.is_file()
]

images = [
    p for p in all_files
    if p.suffix.lower() in image_extensions
]

videos = [
    p for p in all_files
    if p.suffix.lower() in video_extensions
]

print("=== MEDIENÜBERSICHT ===")

print("\nKatzenordner:")
print(len(folders))

print("\nBilder insgesamt:")
print(len(images))

print("\nVideos insgesamt:")
print(len(videos))

print("\nAlle Dateiendungen:")
extensions = Counter(
    p.suffix.lower()
    for p in all_files
)

for ext, count in sorted(extensions.items()):
    print(ext if ext else "[keine Endung]", count)

print("\n=== ORDNERSTATISTIK ===")

folders_with_images = 0
folders_with_videos = 0
folders_with_media = 0
empty_folders = 0

media_counts = []

for folder in folders:

    folder_files = [
        p for p in folder.rglob("*")
        if p.is_file()
    ]

    folder_images = [
        p for p in folder_files
        if p.suffix.lower() in image_extensions
    ]

    folder_videos = [
        p for p in folder_files
        if p.suffix.lower() in video_extensions
    ]

    media_count = len(folder_images) + len(folder_videos)

    media_counts.append(media_count)

    if folder_images:
        folders_with_images += 1

    if folder_videos:
        folders_with_videos += 1

    if media_count > 0:
        folders_with_media += 1
    else:
        empty_folders += 1

print("Ordner mit Bildern:", folders_with_images)
print("Ordner mit Videos:", folders_with_videos)
print("Ordner mit mindestens einem Medium:", folders_with_media)
print("Ordner ohne erkannte Medien:", empty_folders)

if media_counts:
    print("\nMedien pro Ordner:")
    print("Minimum:", min(media_counts))
    print("Maximum:", max(media_counts))
    print("Durchschnitt:", round(sum(media_counts) / len(media_counts), 2))
    