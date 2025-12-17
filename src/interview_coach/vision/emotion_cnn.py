from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class FaceEmotionClassifier(Protocol):
    labels: tuple[str, ...]

    def predict(self, face_bgr: np.ndarray) -> tuple[str, dict[str, float]]:
        ...


@dataclass(frozen=True)
class NullFaceEmotionClassifier:
    labels: tuple[str, ...] = ("neutral", "happy", "sad", "angry", "surprise", "fear", "disgust")

    def predict(self, face_bgr: np.ndarray) -> tuple[str, dict[str, float]]:
        scores = {k: 0.0 for k in self.labels}
        scores["neutral"] = 1.0
        return "neutral", scores


class OnnxFaceEmotionClassifier:
    labels = ("neutral", "happy", "sad", "angry", "surprise", "fear", "disgust")

    def __init__(self, model_path: Path):
        self._model_path = Path(model_path)
        self._session = None
        self._input_name = None

        try:
            import onnxruntime as ort  # type: ignore

            self._session = ort.InferenceSession(
                str(self._model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception:
            self._session = None
            self._input_name = None

    @property
    def available(self) -> bool:
        return self._session is not None and self._input_name is not None

    def predict(self, face_bgr: np.ndarray) -> tuple[str, dict[str, float]]:
        if not self.available:
            return NullFaceEmotionClassifier().predict(face_bgr)

        import cv2  # type: ignore

        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # [1, 3, 64, 64]
        out = self._session.run(None, {self._input_name: x})[0]
        out = np.asarray(out).reshape(-1).astype(np.float32)
        probs = _softmax(out)
        scores = {self.labels[i]: float(probs[i]) for i in range(min(len(self.labels), probs.shape[0]))}
        label = max(scores, key=scores.get) if scores else "neutral"
        return label, scores


def load_face_emotion_classifier(model_path: Path) -> FaceEmotionClassifier:
    model_path = Path(model_path)
    if model_path.exists():
        clf = OnnxFaceEmotionClassifier(model_path)
        if clf.available:
            return clf
    return NullFaceEmotionClassifier()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    denom = np.sum(ex) + 1e-8
    return ex / denom

