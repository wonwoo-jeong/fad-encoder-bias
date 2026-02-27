#!/usr/bin/env python3
"""
Figure 3: Recall — Full pitch-shift trajectory (-8 to +8 st).

Usage:  python figures/gen_fig_diversity.py [--input path/to/fad_results.json]
Output: figures/output/fig_diversity.pdf + fig_diversity.png
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


PITCH_CONDS = [
    ("PitchShift_-8st", -8),
    ("PitchShift_-4st", -4),
    ("PitchShift_-2st", -2),
    ("PitchShift_-1st", -1),
    ("PitchShift_+1st",  1),
    ("PitchShift_+2st",  2),
    ("PitchShift_+4st",  4),
    ("PitchShift_+8st",  8),
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

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    ax.axvspan(2.5, 5.5, color="#E0E0E0", alpha=0.35, zorder=0)
    ax.text(5.6, 0.72, "Recall\nZone",
            fontsize=12, fontstyle="italic", fontweight="bold",
            color="#666666", ha="left", va="center")

    for enc in DRAW_ORDER:
        c, ls, mk, lw, ms, lb = CFG[enc]
        y = []
        for key, _ in PITCH_CONDS:
            vals = [s_norm(get_fad(data, enc, key, ds), fad_max[enc])
                    for ds in DATASETS if get_fad(data, enc, key, ds) is not None]
            y.append(np.mean(vals) if vals else 0)
        zo = 5 if enc in HIGHLIGHT else 3
        ax.plot(range(len(y)), y, color=c, linestyle=ls, linewidth=lw,
                marker=mk, markersize=ms, label=lb, zorder=zo,
                markeredgewidth=0.5, markeredgecolor="white")

    xticks = [str(v) for _, v in PITCH_CONDS]
    ax.set_xticks(range(len(PITCH_CONDS)))
    ax.set_xticklabels(xticks, fontsize=12)
    ax.set_ylabel(r"$S_{\mathrm{norm}}$", fontsize=14)
    ax.set_xlabel("Pitch shift (semitones)", fontsize=14)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="y", labelsize=12)

    handles, labels = ax.get_legend_handles_labels()
    lbl_map = {l: h for h, l in zip(handles, labels)}
    oh = [lbl_map[CFG[e][5]] for e in LEGEND_ORDER]
    ol = [CFG[e][5] for e in LEGEND_ORDER]
    ax.legend(oh, ol, loc="upper left", ncol=2, fontsize=11, framealpha=0.9)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig_diversity.pdf",
                bbox_inches="tight", pad_inches=0.03, dpi=300)
    fig.savefig(FIGURES_DIR / "fig_diversity.png",
                bbox_inches="tight", pad_inches=0.03, dpi=150)
    plt.close(fig)
    print("fig_diversity.pdf / fig_diversity.png saved")


if __name__ == "__main__":
    main()
