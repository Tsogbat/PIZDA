from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


def rms_energy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def estimate_pitch_hz_autocorr(
    x: np.ndarray,
    sample_rate_hz: int,
    fmin_hz: float = 70.0,
    fmax_hz: float = 350.0,
) -> float | None:
    x = np.asarray(x, dtype=np.float32)
    if x.size < int(sample_rate_hz * 0.1):
        return None

    x = x - np.mean(x)
    x = x * np.hamming(x.size).astype(np.float32)

    corr = np.correlate(x, x, mode="full")[x.size - 1 :]
    corr0 = float(corr[0])
    corr[0] = 0.0

    min_lag = int(sample_rate_hz / max(1e-6, fmax_hz))
    max_lag = int(sample_rate_hz / max(1e-6, fmin_hz))
    if max_lag <= min_lag + 2 or max_lag >= corr.size:
        return None

    segment = corr[min_lag:max_lag]
    peak = int(np.argmax(segment)) + min_lag
    if peak <= 0:
        return None

    r = float(corr[peak] / (corr0 + 1e-6))
    if r < 0.2:
        return None

    return float(sample_rate_hz / peak)


def log_mel_spectrogram(
    x: np.ndarray,
    sample_rate_hz: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 40,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if fmax is None:
        fmax = sample_rate_hz / 2.0

    if x.size < win_length:
        x = np.pad(x, (0, win_length - x.size), mode="constant")

    window = np.hanning(win_length).astype(np.float32)
    frames = 1 + max(0, (x.size - win_length) // hop_length)
    spec = np.empty((n_fft // 2 + 1, frames), dtype=np.float32)

    for i in range(frames):
        start = i * hop_length
        frame = x[start : start + win_length]
        if frame.size < win_length:
            frame = np.pad(frame, (0, win_length - frame.size), mode="constant")
        frame = frame * window
        fft = np.fft.rfft(frame, n=n_fft)
        spec[:, i] = (np.abs(fft) ** 2).astype(np.float32)

    mel_fb = _mel_filterbank(sample_rate_hz, n_fft, n_mels, fmin, fmax)
    mel_spec = np.dot(mel_fb, spec)
    mel_spec = np.maximum(mel_spec, 1e-10)
    return np.log(mel_spec).astype(np.float32)


def mfcc(
    x: np.ndarray,
    sample_rate_hz: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    target_frames: int | None = None,
) -> np.ndarray:
    mel = log_mel_spectrogram(x, sample_rate_hz, n_mels=n_mels)
    coeffs = dct(mel, axis=0, norm="ortho")[:n_mfcc, :]
    if target_frames is not None:
        coeffs = _pad_or_trim(coeffs, target_frames)
    return coeffs.astype(np.float32)


def _pad_or_trim(x: np.ndarray, target_frames: int) -> np.ndarray:
    if x.shape[1] == target_frames:
        return x
    if x.shape[1] > target_frames:
        return x[:, :target_frames]
    pad = target_frames - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate_hz: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sample_rate_hz / 2.0, n_freqs)

    mel_min = _hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mel_max = _hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    mels = np.linspace(mel_min, mel_max, n_mels + 2).astype(np.float32)
    hz = _mel_to_hz(mels)

    bins = np.floor((n_fft + 1) * hz / sample_rate_hz).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        if f_m_minus == f_m or f_m == f_m_plus:
            continue
        for k in range(f_m_minus, f_m):
            fb[m - 1, k] = (k - f_m_minus) / float(f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fb[m - 1, k] = (f_m_plus - k) / float(f_m_plus - f_m)

    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    fb *= enorm[:, None].astype(np.float32)
    return fb
