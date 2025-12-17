from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    # Official MediaPipe model (download separately) used for landmarks + blendshapes.
    face_landmarker_task: Path = Path("models/face_landmarker.task")
    face_emotion_onnx: Path = Path("models/face_emotion.onnx")
    vosk_model_dir: Path = Path("models/vosk")


@dataclass(frozen=True)
class VisionConfig:
    camera_index: int = 0
    display_fps: int = 30
    analysis_fps: int = 30
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    yaw_eye_contact_deg: float = 12.0
    pitch_eye_contact_deg: float = 10.0
    emotion_every_n_frames: int = 2
    mirror_preview: bool = True


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int = 16000
    channels: int = 1
    chunk_ms: int = 200
    pause_silence_ms: int = 700
    vad_energy_threshold: float = 0.02
    agc_enabled: bool = True
    agc_target_rms: float = 0.03
    agc_max_gain: float = 8.0
    filler_words: tuple[str, ...] = (
        "uh",
        "um",
        "er",
        "oh",
        "huh",
        "hm",
        "hmm",
        "like",
        "so",
        "well",
        "you know",
        "i mean",
        "literally",
        "actually",
        "basically",
        "seriously",
        "kind of",
        "kinda",
        "sort of",
        "sorta",
        "all right",
        "alright",
        "okay",
        "ok",
        "ah",
        "mhm",
        "nah",
        "anyway",
        "just",
        "totally",
        "honestly",
        "personally",
        "clearly",
        "evidently",
        "you see",
        "i guess",
        "i suppose",
        "or something",
        "i think",
        "i feel",
        "in a way",
        "you know what i mean",
        "at the end of the day",
        "to be honest",
        "needless to say",
        "for what it's worth",
        "truthfully",
        "exactly",
        "certainly",
        "i'm guessing",
        "let me see",
        "let's see",
        "let me think",
        "gosh",
        "jeez",
        "wow",
        "right",
        "yeah",
        "no",
        "em",
        "erm",
        "uhu",
        "eh",
    )


@dataclass(frozen=True)
class OllamaConfig:
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    timeout_s: float = 2.0
    # Separate timeouts so question generation can be slower without breaking real-time UX.
    question_timeout_s: float = 30.0
    speech_timeout_s: float = 6.0
    min_interval_s: float = 3.0


@dataclass(frozen=True)
class InterviewConfig:
    use_llm_questions: bool = True
    num_questions: int = 8
    target_role: str = "Data analyst"


@dataclass(frozen=True)
class TTSConfig:
    enabled: bool = True
    rate_wpm: int = 185
    voice: str | None = None


@dataclass(frozen=True)
class FusionConfig:
    w_eye_contact: float = 0.30
    w_face_emotion: float = 0.15
    w_speech_rate: float = 0.15
    w_fillers: float = 0.15
    w_pauses: float = 0.10
    w_speech_emotion: float = 0.15


@dataclass(frozen=True)
class AppConfig:
    models: ModelPaths = ModelPaths()
    vision: VisionConfig = VisionConfig()
    audio: AudioConfig = AudioConfig()
    ollama: OllamaConfig = OllamaConfig()
    interview: InterviewConfig = InterviewConfig()
    tts: TTSConfig = TTSConfig()
    fusion: FusionConfig = FusionConfig()
