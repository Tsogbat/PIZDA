# Design Notes

## Architecture

- Camera → `VisionWorker` → `VisionAnalyzer` (MediaPipe FaceLandmarker `.task` → landmarks + blendshapes → head pose + eye-contact proxy + emotion mapping; FaceMesh/ONNX fallback)
- Microphone → `AudioWorker` (chunking + energy VAD, optional Vosk STT, prosody + speech emotion from heuristics and optional local Ollama refinement)
- Fusion → `FusionEngine` (late fusion of explainable indicators → confidence score 0–100)
- UI → `PyQt6` dashboard (camera preview, KPIs, transcript, confidence gauge + trend, in-app coaching toasts)
- Session → `SessionRecorder` (time-series samples, question events, transcript; exports JSON/CSV)

## Multimodal Fusion (Explainable)

The confidence score is a weighted sum of normalized indicator scores (0–1), normalized by total weight, then mapped to 0–100.

Indicators:

- Eye contact
- Facial emotion
- Speech rate
- Filler rate
- Long pauses
- Speech emotion

Each indicator stores a human-readable `detail` and a contribution value so the score is explainable and debuggable.

## Performance Techniques

- Frame skipping / decimation: separate display FPS and analysis FPS; emotion inference every N frames.
- Asynchronous processing: vision and audio pipelines run in background threads.
- Model portability: MediaPipe `.task` models run locally; optional local LLM refinement uses your installed Ollama model.
