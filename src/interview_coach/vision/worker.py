from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from interview_coach.config import ModelPaths
from interview_coach.config import VisionConfig
from interview_coach.vision.analyzer import VisionAnalyzer, VisionResult


@dataclass(frozen=True)
class VisionFrame:
    frame_bgr: np.ndarray
    result: VisionResult


class VisionWorker:
    def __init__(self, config: VisionConfig, models: ModelPaths):
        self._cfg = config
        self._analyzer = VisionAnalyzer(config, models)

        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: VisionFrame | None = None
        self._error: str | None = None

        self._analysis_period_s = 1.0 / max(1.0, float(self._cfg.analysis_fps))
        self._display_period_s = 1.0 / max(1.0, float(self._cfg.display_fps))

    @property
    def available(self) -> bool:
        return self._analyzer.available

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        with self._lock:
            self._latest = None
            self._error = None
        self._thread = threading.Thread(target=self._run, name="VisionWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def latest(self) -> VisionFrame | None:
        with self._lock:
            return self._latest

    def reset(self) -> None:
        with self._lock:
            self._latest = None
            self._error = None

    def error(self) -> str | None:
        with self._lock:
            return self._error

    def _run(self) -> None:
        import cv2  # type: ignore

        self._cap = _open_camera_capture(cv2, self._cfg.camera_index)
        if self._cap is None or not self._cap.isOpened():
            with self._lock:
                self._error = (
                    "Camera not available. On macOS: grant Camera permission to your Terminal/Python, "
                    "close other apps using the camera, then restart the app."
                )
            return

        next_analysis = 0.0
        next_display = 0.0
        last_result: VisionResult | None = None

        while not self._stop.is_set():
            now = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._error = "Camera read failed (device busy or permission issue)."
                time.sleep(0.01)
                continue

            if self._cfg.mirror_preview:
                frame = cv2.flip(frame, 1)

            if now >= next_analysis or last_result is None:
                last_result = self._analyzer.process(frame)
                next_analysis = now + self._analysis_period_s

            if now >= next_display:
                with self._lock:
                    self._latest = VisionFrame(frame_bgr=frame, result=last_result)
                next_display = now + self._display_period_s

            sleep_s = min(next_display, next_analysis) - time.perf_counter()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.01))


def _open_camera_capture(cv2, preferred_index: int):
    # macOS often works best with AVFoundation explicitly.
    backends: list[int | None] = []
    if sys.platform == "darwin":
        backends = [getattr(cv2, "CAP_AVFOUNDATION", None), None]
    else:
        backends = [None]

    indices = [preferred_index] + [i for i in range(0, 4) if i != preferred_index]

    for backend in backends:
        for idx in indices:
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            # Hint a reasonable default; some cameras won't honor this.
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            except Exception:
                pass

            # Warm up a few reads to avoid first-frame failure.
            ok = False
            for _ in range(8):
                ok, frame = cap.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    return cap
                time.sleep(0.03)

            try:
                cap.release()
            except Exception:
                pass

    return None
