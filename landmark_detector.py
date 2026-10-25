from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os

import cv2
import numpy as np
from huggingface_hub import hf_hub_download


MODEL_REPO = "hugocornellier/cat-face-landmarks"
MODEL_FILE = "cat_face_landmarks_full.tflite"
INPUT_SIZE = 384


@lru_cache(maxsize=1)
def _load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
    )

    try:
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.cpu_options import CpuOptions
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import Options

        options = Options(
            HardwareAccelerator.CPU,
            cpu_options=CpuOptions(num_threads=max(1, os.cpu_count() or 1)),
        )

        model = CompiledModel.from_file(
            model_path,
            options=options,
        )

        return ("compiled", model)

    except Exception:
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return ("interpreter", interpreter)


def _prepare_face(face_bgr: np.ndarray) -> np.ndarray:
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Empty face crop.")

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return resized.astype(np.float32) / 255.0


def predict_landmarks(face_bgr: np.ndarray) -> np.ndarray:
    """
    Return CatFLW landmarks with shape (48, 2), normalized to [0, 1]
    relative to the supplied face crop.
    """
    mode, model = _load_model()
    image = _prepare_face(face_bgr)
    batch = image[None, ...]

    if mode == "compiled":
        inputs = model.create_input_buffers(0)
        outputs = model.create_output_buffers(0)

        inputs[0].write(batch.astype(np.float32))
        model.run_by_index(0, inputs, outputs)

        flat = outputs[0].read(96, np.float32)
        points = np.asarray(flat, dtype=np.float32).reshape(48, 2)

    else:
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.set_tensor(
            input_details[0]["index"],
            batch.astype(np.float32),
        )
        model.invoke()

        raw = model.get_tensor(output_details[0]["index"])
        points = np.asarray(raw, dtype=np.float32).reshape(48, 2)

    return np.clip(points, 0.0, 1.0)


def landmarks_to_pixels(
    landmarks_xy: np.ndarray,
    face_bgr: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized crop-relative landmarks to crop pixel coordinates.
    """
    height, width = face_bgr.shape[:2]

    points = np.asarray(landmarks_xy, dtype=np.float32).copy()
    points[:, 0] *= width
    points[:, 1] *= height

    return points


def draw_landmarks(
    face_bgr: np.ndarray,
    landmarks_xy: np.ndarray,
) -> np.ndarray:
    output = face_bgr.copy()
    points = landmarks_to_pixels(landmarks_xy, face_bgr)

    for index, (x, y) in enumerate(points):
        cv2.circle(
            output,
            (int(round(x)), int(round(y))),
            2,
            (0, 255, 0),
            -1,
        )

        if index in {0, 2, 4, 8, 12, 13, 22, 27, 36, 39, 46, 47}:
            cv2.putText(
                output,
                str(index),
                (int(round(x)) + 3, int(round(y)) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return output
