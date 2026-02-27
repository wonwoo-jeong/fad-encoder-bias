#!/usr/bin/env python3
"""
Figure 2: Precision — Raw FAD (a, linear) + Normalized S_norm (b).

Usage:  python figures/gen_fig_fidelity.py [--input path/to/fad_results.json]
Output: figures/output/fig_fidelity.pdf + fig_fidelity.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, FIGURES_DIR, ENCODER_KEYS, DATASETS
from compute_fad import s_norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(path):
    with open(path) as f:
        return json.load(f)


def get_fad(data, encoder, label, dataset):
    for d in data:
        if d["encoder"] == encoder and d["label"] == label and d["dataset"] == dataset:
            return d["fad"]
    return None


NOISE_CONDS = [
    ("WhiteNoise_SNR=60dB", "60"),
    ("WhiteNoise_SNR=40dB", "40"),
    ("WhiteNoise_SNR=20dB", "20"),
    ("WhiteNoise_SNR=10dB", "10"),
    ("WhiteNoise_SNR=0dB",  "0"),
    ("WhiteNoise_SNR=-5dB", "\u22125"),
]

CFG = {
    "audiomae": ("#1f77b4", "-",  "o", 2.5, 8,  "AudioMAE"),
    "encodec":  ("#ff7f0e", "-",  "v", 2.0, 7,  "EnCodec"),
    "wav2vec":  ("#8c564b", "--", "*", 2.0, 10, "Wav2Vec 2.0"),
    "vggish":   ("#2ca02c", ":",  "^", 2.5, 8,  "VGGish"),
    "clap":     ("#9467bd", "-.", "D", 2.0, 7,  "CLAP"),
    "whisper":  ("#d62728", "--", "s", 2.5, 8,  "Whisper"),
}

DRAW_ORDER   = ["clap", "wav2vec", "encodec", "vggish", "audiomae", "whisper"]
LEGEND_ORDER = ["audiomae", "encodec", "wav2vec", "vggish", "clap", "whisper"]
HIGHLIGHT    = ["audiomae", "whisper", "vggish"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()
    input_path = Path(args.input) if args.input else RESULTS_DIR / "fad_results.json"
    data = load_data(input_path)

    fad_max = {}
    for enc in ENCODER_KEYS:
        vals = [d["fad"] for d in data if d["encoder"] == enc]
        fad_max[enc] = max(vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # (a) Raw FAD
    for enc in DRAW_ORDER:
        c, ls, mk, lw, ms, lb = CFG[enc]
        y = []
        for key, _ in NOISE_CONDS:
            vals = [get_fad(data, enc, key, ds) for ds in DATASETS
                    if get_fad(data, enc, key, ds) is not None]
            y.append(np.mean(vals) if vals else 0)
        zo = 5 if enc in HIGHLIGHT else 3
        ax1.plot(range(len(y)), y, color=c, linestyle=ls, linewidth=lw,
                 marker=mk, markersize=ms, label=lb, zorder=zo,
                 markeredgewidth=0.5, markeredgecolor="white")

    ax1.set_xticks(range(len(NOISE_CONDS)))
    ax1.set_xticklabels([t for _, t in NOISE_CONDS], fontsize=12)
    ax1.set_ylabel("FAD", fontsize=14)
    ax1.set_xlabel("SNR (dB)", fontsize=14)
    ax1.set_title("(a) Raw FAD", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="y", labelsize=12)

    # (b) Normalized S_norm
    for enc in DRAW_ORDER:
        c, ls, mk, lw, ms, lb = CFG[enc]
        y = []
        for key, _ in NOISE_CONDS:
            vals = [s_norm(get_fad(data, enc, key, ds), fad_max[enc])
                    for ds in DATASETS if get_fad(data, enc, key, ds) is not None]
            y.append(np.mean(vals) if vals else 0)
        zo = 5 if enc in HIGHLIGHT else 3
        ax2.plot(range(len(y)), y, color=c, linestyle=ls, linewidth=lw,
                 marker=mk, markersize=ms, label=lb, zorder=zo,
                 markeredgewidth=0.5, markeredgecolor="white")

    ax2.set_xticks(range(len(NOISE_CONDS)))
    ax2.set_xticklabels([t for _, t in NOISE_CONDS], fontsize=12)
    ax2.set_ylabel(r"$S_{\mathrm{norm}}$", fontsize=14)
    ax2.set_xlabel("SNR (dB)", fontsize=14)
    ax2.set_title("(b) Normalized", fontsize=14, fontweight="bold")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="y", labelsize=12)

    # Legend
    handles, labels = ax1.get_legend_handles_labels()
    lbl_map = {l: h for h, l in zip(handles, labels)}
    oh = [lbl_map[CFG[e][5]] for e in LEGEND_ORDER]
    ol = [CFG[e][5] for e in LEGEND_ORDER]
    fig.legend(oh, ol, loc="lower center", ncol=6,
               fontsize=11, frameon=False, columnspacing=0.8,
               handletextpad=0.4, handlelength=1.8,
               bbox_to_anchor=(0.54, -0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig_fidelity.pdf",
                bbox_inches="tight", pad_inches=0.03, dpi=300)
    fig.savefig(FIGURES_DIR / "fig_fidelity.png",
                bbox_inches="tight", pad_inches=0.03, dpi=150)
    plt.close(fig)
    print("fig_fidelity.pdf / fig_fidelity.png saved")


if __name__ == "__main__":
    main()
