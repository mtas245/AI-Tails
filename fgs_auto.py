from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"

FGS_PROMPTS = {
    "ears": [
        "a cat with ears facing forward in a relaxed position",
        "a cat with ears slightly pulled apart",
        "a cat with ears rotated outward and flattened sideways",
    ],
    "eyes": [
        "a cat with eyes fully open",
        "a cat with eyes partially closed",
        "a cat with strongly squinted eyes",
    ],
    "muzzle": [
        "a cat with a relaxed round muzzle",
        "a cat with mild tension in the muzzle",
        "a cat with a tense elliptical muzzle",
    ],
    "whiskers": [
        "a cat with loose relaxed curved whiskers",
        "a cat with slightly straight whiskers closer together",
        "a cat with straight whiskers projecting forward away from the face",
    ],
    "head": [
        "a cat holding its head above the shoulder line",
        "a cat holding its head aligned with the shoulder line",
        "a cat holding its head below the shoulder line with the chin lowered",
    ],
}

DISPLAY_NAMES = {
    "ears": "Ohrenposition",
    "eyes": "Augenverengung",
    "muzzle": "Muzzle-/Maulspannung",
    "whiskers": "Schnurrhaare",
    "head": "Kopfposition",
}


@lru_cache(maxsize=1)
def load_fgs_model():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def _score_action_unit(model, processor, image: Image.Image, prompts: list[str]):
    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()

    expected_score = float(
        probabilities[0] * 0.0
        + probabilities[1] * 1.0
        + probabilities[2] * 2.0
    )

    discrete_score = int(np.argmax(probabilities))

    return {
        "score": discrete_score,
        "expected_score": expected_score,
        "probabilities": probabilities.tolist(),
    }


def estimate_fgs(face_images: list[Image.Image]):
    if not face_images:
        raise ValueError("No face images supplied.")

    processor, model = load_fgs_model()

    per_image = []

    for image in face_images:
        image_result = {}

        for unit, prompts in FGS_PROMPTS.items():
            image_result[unit] = _score_action_unit(
                model,
                processor,
                image.convert("RGB"),
                prompts,
            )

        per_image.append(image_result)

    session = {}

    for unit in FGS_PROMPTS:
        expected_scores = [
            item[unit]["expected_score"]
            for item in per_image
        ]

        mean_score = float(np.mean(expected_scores))

        session[unit] = {
            "expected_score": mean_score,
            "score": int(np.clip(np.rint(mean_score), 0, 2)),
        }

    total = int(sum(item["score"] for item in session.values()))
    continuous_total = float(
        sum(item["expected_score"] for item in session.values())
    )

    return {
        "action_units": session,
        "total": total,
        "continuous_total": continuous_total,
        "per_image": per_image,
    }
