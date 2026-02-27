#!/usr/bin/env python3
"""
Figure 4: Diverging bar chart (Butterfly) — Structural vs Semantic sensitivity.

Usage:  python figures/gen_fig_blindspots.py [--input path/to/fad_results.json]
Output: figures/output/fig_blindspots.pdf + fig_blindspots.png
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
import matplotlib.patches as mpatches


def load_data(path):
    with open(path) as f:
        return json.load(f)


def get_fad(data, encoder, label, dataset):
    for d in data:
        if d["encoder"] == encoder and d["label"] == label and d["dataset"] == dataset:
            return d["fad"]
    return None


def avg_snorm(data, fad_max, enc, label):
    vals = [s_norm(get_fad(data, enc, label, ds), fad_max[enc])
            for ds in DATASETS if get_fad(data, enc, label, ds) is not None]
    return np.mean(vals) if vals else 0


COLORS = {
    "audiomae": "#1f77b4", "encodec": "#ff7f0e", "wav2vec": "#8c564b",
    "vggish": "#2ca02c", "clap": "#9467bd", "whisper": "#d62728",
}
LABELS = {
    "audiomae": "AudioMAE", "encodec": "EnCodec", "wav2vec": "Wav2Vec 2.0",
    "vggish": "VGGish", "clap": "CLAP", "whisper": "Whisper",
}


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

    enc_data = {}
    for enc in ENCODER_KEYS:
        rev  = avg_snorm(data, fad_max, enc, "TimeReversal")
        shuf = avg_snorm(data, fad_max, enc, "Shuffle_CF_100ms")
        pit  = avg_snorm(data, fad_max, enc, "PitchShift_+8st")
        fmt  = avg_snorm(data, fad_max, enc, "FormantShift_1.4x")
        struct_avg = (rev + shuf) / 2
        sem_avg = (pit + fmt) / 2
        enc_data[enc] = {
            "rev": rev, "shuf": shuf, "pit": pit, "fmt": fmt,
            "struct_avg": struct_avg, "sem_avg": sem_avg,
            "diff": struct_avg - sem_avg,
        }

    sorted_encs = sorted(ENCODER_KEYS, key=lambda e: enc_data[e]["diff"])

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bar_h = 0.32
    gap = 0.06
    y_positions = np.arange(len(sorted_encs))

    for i, enc in enumerate(sorted_encs):
        d = enc_data[enc]
        c = COLORS[enc]
        y = y_positions[i]

        ax.barh(y + gap / 2, -d["rev"], height=bar_h, color=c, alpha=0.90,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax.barh(y - gap / 2 - bar_h, -d["shuf"], height=bar_h, color=c,
                alpha=0.50, edgecolor="white", linewidth=0.5, zorder=3,
                hatch="///")

        ax.barh(y + gap / 2, d["pit"], height=bar_h, color=c, alpha=0.90,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax.barh(y - gap / 2 - bar_h, d["fmt"], height=bar_h, color=c,
                alpha=0.50, edgecolor="white", linewidth=0.5, zorder=3,
                hatch="///")

    ax.axvline(x=0, color="#333333", linewidth=1.5, zorder=5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([LABELS[e] for e in sorted_encs],
                       fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel(r"$S_{\mathrm{norm}}$", fontsize=14)

    ax.text(-0.52, len(sorted_encs) + 0.1,
            r"$\longleftarrow$ Structural Alignment",
            fontsize=12, fontweight="bold", ha="center", va="bottom",
            color="#444444")
    ax.text(0.52, len(sorted_encs) + 0.1,
            r"Semantic Alignment $\longrightarrow$",
            fontsize=12, fontweight="bold", ha="center", va="bottom",
            color="#444444")

    ax.set_xlim(-1.12, 1.0)
    ax.set_ylim(-0.7, len(sorted_encs) - 0.3)

    xticks_abs = [1.0, 0.75, 0.5, 0.25, 0, 0.25, 0.5, 0.75, 1.0]
    xtick_pos  = [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([f"{v:.2f}" for v in xticks_abs], fontsize=11)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    leg_dark = mpatches.Patch(facecolor="#888888", alpha=0.90,
                              label="Reversal / Pitch +8 st")
    leg_light = mpatches.Patch(facecolor="#888888", alpha=0.50, hatch="///",
                               label="Shuffle 100 ms / Formant 1.4\u00d7")
    fig.legend(handles=[leg_dark, leg_light], loc="lower center",
               ncol=2, fontsize=11.5, frameon=False,
               handletextpad=0.4, columnspacing=1.5,
               bbox_to_anchor=(0.6, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig_blindspots.pdf",
                bbox_inches="tight", pad_inches=0.03, dpi=300)
    fig.savefig(FIGURES_DIR / "fig_blindspots.png",
                bbox_inches="tight", pad_inches=0.03, dpi=150)
    plt.close(fig)
    print("fig_blindspots.pdf / fig_blindspots.png saved")


if __name__ == "__main__":
    main()
