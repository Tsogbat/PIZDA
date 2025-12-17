from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from interview_coach.config import VisionConfig
from interview_coach.utils.smoothing import EWMA, clamp
from interview_coach.utils.timing import time_call
from interview_coach.config import ModelPaths
from interview_coach.vision.emotion_cnn import FaceEmotionClassifier, load_face_emotion_classifier


@dataclass(frozen=True)
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class VisionResult:
    timestamp_s: float
    face_detected: bool
    eye_contact: float  # 0..1
    head_pose: HeadPose | None
    emotion: str
    emotion_scores: dict[str, float]
    latency_ms: float


class VisionAnalyzer:
    def __init__(self, config: VisionConfig, models: ModelPaths):
        self._cfg = config
        self._models = models
        self._mp_face_mesh = None
        self._face_mesh = None
        self._emotion = load_face_emotion_classifier(models.face_emotion_onnx)

        self._mp = None
        self._landmarker = None
        self._use_landmarker = False
        self._landmarker_t0 = time.perf_counter()

        self._eye_ewma = EWMA(alpha=0.35)
        self._yaw_ewma = EWMA(alpha=0.25)
        self._pitch_ewma = EWMA(alpha=0.25)
        self._roll_ewma = EWMA(alpha=0.25)

        self._frame_count = 0
        self._last_emotion: tuple[str, dict[str, float]] = ("neutral", {"neutral": 1.0})

        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        # Prefer MediaPipe Tasks FaceLandmarker (supports blendshapes) when available.
        task_path = getattr(self._models, "face_landmarker_task", None)
        candidates = []
        if task_path is not None:
            candidates.append(task_path)
        candidates.extend(
            [
                ModelPaths().face_landmarker_task,
                ModelPaths().face_landmarker_task.with_name("face_landmarker.task"),
                ModelPaths().face_landmarker_task.with_name("face_landmarker_with_blendshapes.task"),
            ]
        )
        chosen = next((p for p in candidates if p.exists()), None)

        if chosen is not None and chosen.exists():
            try:
                import mediapipe as mp  # type: ignore
                from mediapipe.tasks.python import vision  # type: ignore
                from mediapipe.tasks.python.core import base_options  # type: ignore

                options = vision.FaceLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=str(chosen)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
                self._mp = mp
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
                self._use_landmarker = True
                return
            except Exception:
                self._mp = None
                self._landmarker = None
                self._use_landmarker = False

        try:
            import mediapipe as mp  # type: ignore

            self._mp_face_mesh = mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._cfg.min_detection_confidence,
                min_tracking_confidence=self._cfg.min_tracking_confidence,
            )
        except Exception:
            self._mp_face_mesh = None
            self._face_mesh = None

    @property
    def available(self) -> bool:
        return self._use_landmarker or self._face_mesh is not None

    def process(self, frame_bgr: np.ndarray) -> VisionResult:
        ts = time.time()
        (tb, out) = time_call(self._process_impl, frame_bgr)
        result: VisionResult = out  # type: ignore[assignment]
        return VisionResult(
            timestamp_s=ts,
            face_detected=result.face_detected,
            eye_contact=result.eye_contact,
            head_pose=result.head_pose,
            emotion=result.emotion,
            emotion_scores=result.emotion_scores,
            latency_ms=tb.duration_ms,
        )

    def _process_impl(self, frame_bgr: np.ndarray) -> VisionResult:
        if self._use_landmarker and self._landmarker is not None and self._mp is not None:
            return self._process_with_landmarker(frame_bgr)

        if self._face_mesh is None:
            return VisionResult(
                timestamp_s=time.time(),
                face_detected=False,
                eye_contact=0.0,
                head_pose=None,
                emotion="neutral",
                emotion_scores={"neutral": 1.0},
                latency_ms=0.0,
            )

        import cv2  # type: ignore

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_out = self._face_mesh.process(rgb)

        if not mp_out.multi_face_landmarks:
            self._eye_ewma.value = None
            return VisionResult(
                timestamp_s=time.time(),
                face_detected=False,
                eye_contact=0.0,
                head_pose=None,
                emotion="neutral",
                emotion_scores={"neutral": 1.0},
                latency_ms=0.0,
            )

        landmarks = mp_out.multi_face_landmarks[0].landmark

        head_pose = _estimate_head_pose_deg(landmarks, w, h)
        if head_pose is not None:
            yaw = self._yaw_ewma.update(head_pose.yaw_deg)
            pitch = self._pitch_ewma.update(head_pose.pitch_deg)
            roll = self._roll_ewma.update(head_pose.roll_deg)
            head_pose = HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)

        eye_contact = _eye_contact_from_pose(head_pose, self._cfg.yaw_eye_contact_deg, self._cfg.pitch_eye_contact_deg)
        eye_contact = self._eye_ewma.update(eye_contact)

        self._frame_count += 1
        if self._frame_count % max(1, self._cfg.emotion_every_n_frames) == 0:
            face = _crop_face(frame_bgr, landmarks, w, h)
            if face is not None:
                self._last_emotion = self._emotion.predict(face)

        emotion, scores = self._last_emotion

        return VisionResult(
            timestamp_s=time.time(),
            face_detected=True,
            eye_contact=clamp(float(eye_contact), 0.0, 1.0),
            head_pose=head_pose,
            emotion=emotion,
            emotion_scores=scores,
            latency_ms=0.0,
        )

    def _process_with_landmarker(self, frame_bgr: np.ndarray) -> VisionResult:
        import cv2  # type: ignore

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - self._landmarker_t0) * 1000.0)

        try:
            result = self._landmarker.detect_for_video(mp_image, ts_ms)
        except Exception:
            result = None

        if result is None or not getattr(result, "face_landmarks", None):
            self._eye_ewma.value = None
            return VisionResult(
                timestamp_s=time.time(),
                face_detected=False,
                eye_contact=0.0,
                head_pose=None,
                emotion="neutral",
                emotion_scores={"neutral": 1.0},
                latency_ms=0.0,
            )

        landmarks = result.face_landmarks[0]
        head_pose = _head_pose_from_transform(result)
        if head_pose is not None:
            yaw = self._yaw_ewma.update(head_pose.yaw_deg)
            pitch = self._pitch_ewma.update(head_pose.pitch_deg)
            roll = self._roll_ewma.update(head_pose.roll_deg)
            head_pose = HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)

        eye_contact = _eye_contact_from_pose(head_pose, self._cfg.yaw_eye_contact_deg, self._cfg.pitch_eye_contact_deg)
        eye_contact = self._eye_ewma.update(eye_contact)

        self._frame_count += 1
        if self._frame_count % max(1, self._cfg.emotion_every_n_frames) == 0:
            emotion, scores = _emotion_from_blendshapes(result)
            self._last_emotion = (emotion, scores)

        emotion, scores = self._last_emotion
        return VisionResult(
            timestamp_s=time.time(),
            face_detected=True,
            eye_contact=clamp(float(eye_contact), 0.0, 1.0),
            head_pose=head_pose,
            emotion=emotion,
            emotion_scores=scores,
            latency_ms=0.0,
        )


def _eye_contact_from_pose(pose: HeadPose | None, yaw_thresh: float, pitch_thresh: float) -> float:
    if pose is None:
        return 0.0
    yaw_ok = max(0.0, 1.0 - (abs(pose.yaw_deg) / max(1e-3, yaw_thresh)))
    pitch_ok = max(0.0, 1.0 - (abs(pose.pitch_deg) / max(1e-3, pitch_thresh)))
    return float(clamp((yaw_ok * pitch_ok) ** 0.5, 0.0, 1.0))


def _crop_face(frame_bgr: np.ndarray, landmarks: list[Any], w: int, h: int) -> np.ndarray | None:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    if not xs or not ys:
        return None
    x0 = int(max(0.0, (min(xs) - 0.05)) * w)
    y0 = int(max(0.0, (min(ys) - 0.08)) * h)
    x1 = int(min(1.0, (max(xs) + 0.05)) * w)
    y1 = int(min(1.0, (max(ys) + 0.08)) * h)
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1].copy()


def _estimate_head_pose_deg(landmarks: list[Any], w: int, h: int) -> HeadPose | None:
    # Mediapipe Face Mesh landmark indices used for a stable head pose estimate.
    # These are standard indices (nose tip, chin, eye corners, mouth corners).
    idx = {
        "nose": 1,
        "chin": 152,
        "l_eye": 33,
        "r_eye": 263,
        "l_mouth": 61,
        "r_mouth": 291,
    }
    try:
        image_points = np.array(
            [
                (landmarks[idx["nose"]].x * w, landmarks[idx["nose"]].y * h),
                (landmarks[idx["chin"]].x * w, landmarks[idx["chin"]].y * h),
                (landmarks[idx["l_eye"]].x * w, landmarks[idx["l_eye"]].y * h),
                (landmarks[idx["r_eye"]].x * w, landmarks[idx["r_eye"]].y * h),
                (landmarks[idx["l_mouth"]].x * w, landmarks[idx["l_mouth"]].y * h),
                (landmarks[idx["r_mouth"]].x * w, landmarks[idx["r_mouth"]].y * h),
            ],
            dtype=np.float64,
        )
    except Exception:
        return None

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # nose tip
            (0.0, -330.0, -65.0),  # chin
            (-225.0, 170.0, -135.0),  # left eye corner
            (225.0, 170.0, -135.0),  # right eye corner
            (-150.0, -150.0, -125.0),  # left mouth
            (150.0, -150.0, -125.0),  # right mouth
        ],
        dtype=np.float64,
    )

    try:
        import cv2  # type: ignore

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
    except Exception:
        return None

    yaw, pitch, roll = _rotation_matrix_to_euler_deg(rmat)
    return HeadPose(yaw_deg=float(yaw), pitch_deg=float(pitch), roll_deg=float(roll))


def _rotation_matrix_to_euler_deg(rmat: np.ndarray) -> tuple[float, float, float]:
    # Returns yaw (Y), pitch (X), roll (Z) in degrees.
    # Convention chosen for stability; used only for eye-contact thresholds.
    sy = float(np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0]))
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0
    return (np.degrees(yaw), np.degrees(pitch), np.degrees(roll))


def _head_pose_from_transform(result: Any) -> HeadPose | None:
    mats = getattr(result, "facial_transformation_matrixes", None)
    if not mats:
        return None
    try:
        m = np.asarray(mats[0], dtype=np.float64)
        if m.shape[0] >= 3 and m.shape[1] >= 3:
            r = m[:3, :3]
        else:
            return None
        yaw, pitch, roll = _rotation_matrix_to_euler_deg(r)
        return HeadPose(yaw_deg=float(yaw), pitch_deg=float(pitch), roll_deg=float(roll))
    except Exception:
        return None


def _emotion_from_blendshapes(result: Any) -> tuple[str, dict[str, float]]:
    labels = ("neutral", "happy", "sad", "angry", "surprise", "fear", "disgust")
    bss = getattr(result, "face_blendshapes", None)
    if not bss:
        return "neutral", {"neutral": 1.0}

    try:
        cats = bss[0]
        bs = {c.category_name: float(c.score) for c in cats}
    except Exception:
        return "neutral", {"neutral": 1.0}

    def g(name: str) -> float:
        return float(bs.get(name, 0.0))

    smile = 0.5 * (g("mouthSmileLeft") + g("mouthSmileRight"))
    cheek = 0.5 * (g("cheekSquintLeft") + g("cheekSquintRight"))
    frown = 0.5 * (g("mouthFrownLeft") + g("mouthFrownRight"))
    brow_down = 0.5 * (g("browDownLeft") + g("browDownRight"))
    brow_up = max(g("browInnerUp"), 0.5 * (g("browOuterUpLeft") + g("browOuterUpRight")))
    jaw_open = g("jawOpen")
    eye_wide = 0.5 * (g("eyeWideLeft") + g("eyeWideRight"))
    eye_squint = 0.5 * (g("eyeSquintLeft") + g("eyeSquintRight"))
    nose_sneer = 0.5 * (g("noseSneerLeft") + g("noseSneerRight"))
    mouth_press = 0.5 * (g("mouthPressLeft") + g("mouthPressRight"))
    mouth_stretch = 0.5 * (g("mouthStretchLeft") + g("mouthStretchRight"))
    upper_up = 0.5 * (g("mouthUpperUpLeft") + g("mouthUpperUpRight"))

    raw = {
        "happy": clamp(0.75 * smile + 0.25 * cheek, 0.0, 1.0),
        "sad": clamp(0.65 * frown + 0.25 * brow_up + 0.10 * (1.0 - eye_wide), 0.0, 1.0),
        "angry": clamp(0.60 * brow_down + 0.20 * eye_squint + 0.20 * mouth_press, 0.0, 1.0),
        "surprise": clamp(0.55 * jaw_open + 0.25 * brow_up + 0.20 * eye_wide, 0.0, 1.0),
        "fear": clamp(0.50 * eye_wide + 0.20 * jaw_open + 0.30 * mouth_stretch, 0.0, 1.0),
        "disgust": clamp(0.55 * nose_sneer + 0.25 * upper_up + 0.20 * mouth_press, 0.0, 1.0),
    }
    raw["neutral"] = clamp(1.0 - max(raw.values() or [0.0]), 0.0, 1.0)

    total = sum(raw.values()) + 1e-8
    scores = {k: float(v / total) for k, v in raw.items()}
    for lab in labels:
        scores.setdefault(lab, 0.0)
    label = max(scores, key=scores.get) if scores else "neutral"
    return label, scores
