"""
Global configuration — paths, constants, and perturbation condition definitions.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
CACHE_DIR   = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures" / "output"

LIBRI_DIR = DATA_DIR / "LibriSpeech" / "test-clean"
ESC50_DIR = DATA_DIR / "ESC-50-master"

for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Preprocessing ────────────────────────────────────────────────────────
TARGET_LUFS = -23.0  # ITU-R BS.1770-4

# ── Encoder registry keys ────────────────────────────────────────────────
ENCODER_KEYS = ["whisper", "clap", "vggish", "encodec", "audiomae", "wav2vec"]

# ── Perturbation condition definitions ───────────────────────────────────
# Each condition: (question, task, params, datasets, label)
def build_conditions():
    """Return the full list of perturbation conditions as used in the paper."""
    conds = []

    BOTH = ["librispeech", "esc50"]

    # ── Recall (within-class variation) ──
    for s in [1, 2, 4, 8]:
        conds.append(dict(q="Q1", task="pitch_shift",
                          params=dict(n_steps=s),
                          datasets=BOTH,
                          label=f"PitchShift_+{s}st"))
    for s in [-1, -2, -4, -8]:
        conds.append(dict(q="Q1", task="pitch_shift",
                          params=dict(n_steps=s),
                          datasets=BOTH,
                          label=f"PitchShift_{s}st"))
    for r in [0.8, 0.9, 1.1, 1.2]:
        conds.append(dict(q="Q1", task="time_stretch",
                          params=dict(rate=r),
                          datasets=BOTH,
                          label=f"TimeStretch_{r}x"))
    for ratio in [1.1, 1.2, 1.3, 1.4]:
        conds.append(dict(q="Q1", task="formant_shift",
                          params=dict(ratio=ratio),
                          datasets=BOTH,
                          label=f"FormantShift_{ratio}x"))

    # ── Precision (signal degradation) ──
    for rt in [0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0]:
        conds.append(dict(q="Q2", task="reverb",
                          params=dict(rt60=rt),
                          datasets=BOTH,
                          label=f"Reverb_RT60={rt}s"))

    # ── Structural Alignment ──
    conds.append(dict(q="Q3", task="time_reverse",
                      params={},
                      datasets=BOTH,
                      label="TimeReversal"))
    for ms in [1000, 500, 250, 100]:
        conds.append(dict(q="Q3", task="shuffle",
                          params=dict(chunk_ms=ms),
                          datasets=BOTH,
                          label=f"Shuffle_CF_{ms}ms"))
    # Legacy shuffle conditions (without crossfade label)
    for ms in [1000, 500, 250]:
        conds.append(dict(q="Q3", task="shuffle",
                          params=dict(chunk_ms=ms),
                          datasets=BOTH,
                          label=f"Shuffle_{ms}ms"))

    # ── Precision (noise / filtering) ──
    for snr in [60, 40, 20, 10, 0, -5]:
        conds.append(dict(q="Q4", task="white_noise",
                          params=dict(snr_db=snr),
                          datasets=BOTH,
                          label=f"WhiteNoise_SNR={snr}dB"))
    for cf in [8000, 6000, 4000, 2000, 1000]:
        conds.append(dict(q="Q4", task="lowpass",
                          params=dict(cutoff=cf),
                          datasets=BOTH,
                          label=f"LowPass_{cf}Hz"))

    return conds


# ── Axis label groups (for score computation) ────────────────────────────
RECALL_LABELS = [
    "PitchShift_+1st", "PitchShift_+2st",
    "PitchShift_-1st", "PitchShift_-2st",
    "TimeStretch_0.9x", "TimeStretch_1.1x",
]

PRECISION_LABELS = [
    "WhiteNoise_SNR=60dB", "WhiteNoise_SNR=40dB", "WhiteNoise_SNR=20dB",
    "WhiteNoise_SNR=10dB", "WhiteNoise_SNR=0dB", "WhiteNoise_SNR=-5dB",
    "LowPass_8000Hz", "LowPass_6000Hz", "LowPass_4000Hz",
    "LowPass_2000Hz", "LowPass_1000Hz",
    "Reverb_RT60=0.1s", "Reverb_RT60=0.2s", "Reverb_RT60=0.25s",
    "Reverb_RT60=0.4s", "Reverb_RT60=0.5s", "Reverb_RT60=0.6s",
    "Reverb_RT60=0.8s", "Reverb_RT60=1.0s", "Reverb_RT60=2.0s",
]

SEMANTIC_LABELS = [
    "PitchShift_+4st", "PitchShift_+8st",
    "PitchShift_-4st", "PitchShift_-8st",
    "FormantShift_1.3x", "FormantShift_1.4x",
]

STRUCTURAL_LABELS = [
    "TimeReversal",
    "Shuffle_CF_1000ms", "Shuffle_CF_500ms",
    "Shuffle_CF_250ms", "Shuffle_CF_100ms",
]

DATASETS = ["librispeech", "esc50"]
