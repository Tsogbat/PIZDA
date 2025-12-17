# Evaluation

This system measures and exports latency for vision, audio, and multimodal fusion during every session. Accuracy is evaluated qualitatively (and can be extended to quantitative evaluation with labeled datasets).

## Latency Measurement

**Where latency is measured**

- Vision latency: `VisionAnalyzer.process()` measures per-frame analysis time and stores it in `VisionResult.latency_ms`.
- Audio latency: `AudioWorker._process_chunk()` measures per-chunk processing time and stores it in `AudioResult.latency_ms`.
- Fusion latency: `FusionEngine.fuse()` measures per-update fusion time and stores it in `FusionResult.latency_ms`.

**Where latency is recorded**

- Each UI tick (rate-limited to ~5 Hz) records a `SessionSample` including `vision_latency_ms`, `audio_latency_ms`, and `fusion_latency_ms`.
- Exports:
  - JSON: `reports/session_YYYYMMDD_HHMMSS.json`
  - CSV: `reports/session_YYYYMMDD_HHMMSS.csv`

**Session summary**

The post-session view computes:

- Average latency per module (ms)
- p95 latency per module (ms)

These are computed from the exported time-series (see `src/interview_coach/analytics.py`).

## Accuracy Analysis (Practical)

This repo is designed to run offline and supports drop-in local models for:

- `models/face_landmarker.task` (MediaPipe FaceLandmarker; landmarks + blendshapes)
- Speech emotion uses prosody heuristics by default and can optionally use a local Ollama model (`OllamaConfig`) for refinement.

If a model is missing, the system continues to run with a neutral fallback label; this preserves end-to-end performance measurement while clearly limiting emotion accuracy.

**Vision**

- Eye contact uses head pose estimated from MediaPipe face outputs (transformation matrix when using FaceLandmarker; PnP fallback when using FaceMesh). This is robust for typical webcam distances but can drift with:
  - Large yaw/pitch, extreme expressions, heavy occlusion, and strong lens distortion.
- Facial emotion accuracy depends on the FaceLandmarker model quality plus the blendshape→emotion mapping and lighting/pose conditions.

**Audio**

- Speech rate and filler metrics depend on transcription quality (Vosk model, background noise, microphone quality).
- Pitch is estimated via autocorrelation on a 1s rolling window and can be unreliable for very quiet speech, overlapping speakers, or noisy environments.
- Speech emotion accuracy depends on the prosody heuristic thresholds and (if enabled) the local Ollama model’s behavior; results are best treated as coaching signals.

## Limitations & Failure Cases

- Multi-person in frame: current pipeline assumes a single face; it uses the first detected face.
- Strong backlight or low light reduces landmark quality and emotion inference.
- Glasses glare or face occlusion reduces landmark stability and head pose accuracy.
- Noisy rooms can trigger false speaking segments and degrade STT.
- Emotion classifiers are highly sensitive to dataset bias; results should be presented as coaching signals, not diagnoses.

## Engineering Trade-offs

- **Latency vs accuracy**: emotion inference is optionally decimated (`VisionConfig.emotion_every_n_frames`) to reduce CPU load while keeping eye-contact estimation responsive.
- **Frame skipping**: the camera preview can run at 30 FPS while analysis runs at the same or lower rate depending on hardware.
- **Asynchronous pipelines**: camera and microphone analysis run in background threads; UI renders the latest available results.
- **Offline-first**: models are local; no network calls are required at runtime.
