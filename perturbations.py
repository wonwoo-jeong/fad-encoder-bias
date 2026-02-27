"""
DSP-based audio perturbation pipeline.

Each function takes (wav, sr, **params) and returns a perturbed waveform tensor.
"""

import math
import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF


# ═══════════════════════════════════════════════════════════════════════════
# Recall axis — within-class variation
# ═══════════════════════════════════════════════════════════════════════════

def perturb_pitch_shift(wav: torch.Tensor, sr: int, n_steps: int) -> torch.Tensor:
    """Pitch shift by n_steps semitones (GPU-accelerated if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav_dev = wav.to(device)
    shifted = torchaudio.functional.pitch_shift(wav_dev, sr, n_steps)
    return shifted.cpu()


def perturb_time_stretch(wav: torch.Tensor, sr: int, rate: float) -> torch.Tensor:
    """Phase-vocoder time stretch (no pitch change)."""
    n_fft, hop = 2048, 512
    for attempt_device in ["cuda", "cpu"]:
        if attempt_device == "cuda" and not torch.cuda.is_available():
            continue
        try:
            device = torch.device(attempt_device)
            wav_1d = wav.squeeze().to(device)
            window = torch.hann_window(n_fft, device=device)
            spec = torch.stft(wav_1d, n_fft=n_fft, hop_length=hop,
                              window=window, return_complex=True)
            n_freq = spec.shape[0]
            phase_advance = torch.linspace(
                0, math.pi * hop, n_freq, device=device
            )[..., None]
            stretched = AF.phase_vocoder(spec, rate, phase_advance)
            out = torch.istft(stretched, n_fft=n_fft, hop_length=hop,
                              window=window)
            result = out.unsqueeze(0).cpu()
            if attempt_device == "cuda":
                del wav_1d, window, spec, phase_advance, stretched, out
                torch.cuda.empty_cache()
            return result
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if attempt_device == "cuda":
                torch.cuda.empty_cache()
                continue
            raise


# ═══════════════════════════════════════════════════════════════════════════
# Precision axis — signal degradation
# ═══════════════════════════════════════════════════════════════════════════

def perturb_white_noise(wav: torch.Tensor, sr: int, snr_db: int) -> torch.Tensor:
    """Additive Gaussian white noise at given SNR (dB)."""
    sig_pw = wav.pow(2).mean()
    noise_pw = sig_pw / (10 ** (snr_db / 10))
    noise = torch.randn_like(wav) * torch.sqrt(noise_pw + 1e-12)
    return wav + noise


def perturb_lowpass(wav: torch.Tensor, sr: int, cutoff: int) -> torch.Tensor:
    """Low-pass biquad filter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = torchaudio.functional.lowpass_biquad(wav.to(device), sr, cutoff)
    return out.cpu()


def perturb_reverb(wav: torch.Tensor, sr: int, rt60: float) -> torch.Tensor:
    """Reverb using pedalboard (wet 50%)."""
    from pedalboard import Pedalboard, Reverb
    room = min(rt60 / 2.5, 1.0)
    board = Pedalboard([Reverb(room_size=room, damping=0.5,
                               wet_level=0.5, dry_level=0.5)])
    out = board(wav.squeeze().numpy(), sr)
    return torch.from_numpy(out).unsqueeze(0).float()


# ═══════════════════════════════════════════════════════════════════════════
# Semantic Alignment axis — identity-altering transformations
# ═══════════════════════════════════════════════════════════════════════════

def perturb_formant_shift(wav: torch.Tensor, sr: int, ratio: float) -> torch.Tensor:
    """Formant shift via Praat Change Gender (parselmouth), preserving F0."""
    import parselmouth
    from parselmouth.praat import call
    audio_np = wav.squeeze().numpy().astype(np.float64)
    snd = parselmouth.Sound(audio_np, sampling_frequency=sr)
    out_snd = call(snd, "Change gender", 75.0, 600.0, ratio, 0.0, 1.0, 1.0)
    out_np = out_snd.values.squeeze()
    return torch.from_numpy(out_np).unsqueeze(0).float()


# ═══════════════════════════════════════════════════════════════════════════
# Structural Alignment axis — temporal structure disruption
# ═══════════════════════════════════════════════════════════════════════════

def perturb_time_reverse(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Full time reversal."""
    return torch.flip(wav, [1])


def perturb_shuffle(wav: torch.Tensor, sr: int,
                    chunk_ms: int, crossfade_ms: int = 10) -> torch.Tensor:
    """Random chunk shuffle with cross-fade to suppress click artifacts."""
    chunk = int(sr * chunk_ms / 1000)
    fade = max(int(sr * crossfade_ms / 1000), 1)

    flat = wav.squeeze()
    pieces = [flat[i:i + chunk] for i in range(0, flat.shape[0], chunk)]
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(pieces))
    shuffled = [pieces[i] for i in idx]

    if fade == 0 or len(shuffled) <= 1:
        return torch.cat(shuffled).unsqueeze(0)

    out_parts = [shuffled[0]]
    for i in range(1, len(shuffled)):
        prev = out_parts[-1]
        curr = shuffled[i]
        actual_fade = min(fade, prev.shape[0], curr.shape[0])
        if actual_fade < 2:
            out_parts.append(curr)
            continue
        fo = torch.linspace(1.0, 0.0, actual_fade)
        fi = torch.linspace(0.0, 1.0, actual_fade)
        overlap = prev[-actual_fade:] * fo + curr[:actual_fade] * fi
        out_parts[-1] = prev[:-actual_fade]
        out_parts.append(overlap)
        out_parts.append(curr[actual_fade:])

    return torch.cat(out_parts).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════

PERTURBATION_FN = {
    "pitch_shift":   perturb_pitch_shift,
    "time_stretch":  perturb_time_stretch,
    "white_noise":   perturb_white_noise,
    "lowpass":       perturb_lowpass,
    "reverb":        perturb_reverb,
    "formant_shift": perturb_formant_shift,
    "time_reverse":  perturb_time_reverse,
    "shuffle":       perturb_shuffle,
}


def apply_perturbation(wav: torch.Tensor, sr: int,
                       task: str, params: dict) -> torch.Tensor:
    """Apply a named perturbation with given parameters."""
    fn = PERTURBATION_FN[task]
    return fn(wav, sr, **params)
