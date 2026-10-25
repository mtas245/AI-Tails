from __future__ import annotations

from pathlib import Path
import importlib
import json
import sys


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    ROOT / "models" / "cat_face_yolov8n.pt",
    ROOT / "models" / "healthy_geometry_baseline.json",
]

OPTIONAL_FILES = [
    ROOT / "models" / "health_classifier.pt",
]

REQUIRED_MODULES = [
    "cv2",
    "numpy",
    "streamlit",
    "torch",
    "torchvision",
    "ultralytics",
    "huggingface_hub",
]

EXPECTED_SCHEMA_VERSION = "ai-tails.vision.v1"


def ok(message: str):
    print(f"[OK]   {message}")


def warn(message: str):
    print(f"[WARN] {message}")


def fail(message: str):
    print(f"[FAIL] {message}")


def main() -> int:
    errors = 0

    print("=== AI-TAILS MVP VALIDATION ===\n")

    for path in REQUIRED_FILES:
        if path.exists():
            ok(f"Required file: {path.relative_to(ROOT)}")
        else:
            fail(f"Missing required file: {path.relative_to(ROOT)}")
            errors += 1

    for path in OPTIONAL_FILES:
        if path.exists():
            ok(f"Optional file: {path.relative_to(ROOT)}")
        else:
            warn(
                f"Optional file missing: {path.relative_to(ROOT)} "
                "(MVP still works without it)"
            )

    print()

    baseline_path = ROOT / "models" / "healthy_geometry_baseline.json"

    if baseline_path.exists():
        try:
            payload = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )

            for key in ("healthy_cats", "mean", "std"):
                if key not in payload:
                    raise ValueError(f"missing key: {key}")

            if not payload["mean"] or not payload["std"]:
                raise ValueError("mean/std must not be empty")

            if set(payload["mean"]) != set(payload["std"]):
                raise ValueError("mean/std feature keys do not match")

            healthy_cats = int(payload["healthy_cats"])
            ok(f"Healthy baseline schema valid ({healthy_cats} cats)")

            if healthy_cats < 10:
                warn(
                    "Healthy baseline is still small; report this as an MVP limitation."
                )

        except Exception as exc:
            fail(f"Healthy baseline invalid: {exc}")
            errors += 1

    print()

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            ok(f"Import: {module_name}")
        except Exception as exc:
            fail(f"Import failed: {module_name} ({exc})")
            errors += 1

    print()

    try:
        from vision_contract import SCHEMA_VERSION

        if SCHEMA_VERSION != EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"expected {EXPECTED_SCHEMA_VERSION}, got {SCHEMA_VERSION}"
            )

        ok(f"Vision contract: {SCHEMA_VERSION}")

    except Exception as exc:
        fail(f"Vision contract invalid: {exc}")
        errors += 1

    try:
        import fgs_geometry
        import landmark_detector

        ok("FGS geometry and landmark modules import successfully")

    except Exception as exc:
        fail(f"Pipeline module import failed: {exc}")
        errors += 1

    print()

    if errors:
        fail(f"MVP validation finished with {errors} blocking issue(s).")
        return 1

    ok("MVP validation passed.")
    print(
        "\nThe app is ready for an end-to-end demo. "
        "Run: python -m streamlit run app.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
