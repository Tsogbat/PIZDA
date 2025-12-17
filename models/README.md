# Models directory

This project is designed to run fully offline once models are available locally.

Place the following optional models here:

- `models/face_landmarker.task` (recommended; `models/face_landmarker_with_blendshapes.task` also works)
  - Official MediaPipe FaceLandmarker model (landmarks + blendshapes), used for:
    - Face landmarks (for head-pose / eye-contact proxy)
    - Facial expression → emotion mapping (happy/sad/angry/etc.)
  - Download: `https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models`

- Speech emotion (no file required)
  - The app estimates speech emotion from prosody (pitch/energy/rate) by default.
  - If you have Ollama installed locally, it can optionally refine speech emotion using your local model (default: `qwen3:8b`).
  - Configure in `src/interview_coach/config.py` (`OllamaConfig`).
  - Quick check: `python3 -m interview_coach.diagnose`

- `models/vosk/`
  - A Vosk language model folder (e.g., `vosk-model-small-en-us-0.15/` unpacked here).
  - Configure the exact path in the app settings if needed.

If a model is missing (e.g., no FaceLandmarker task), the app falls back to lightweight heuristics (still recording latency/metrics), so the HMI remains usable.
