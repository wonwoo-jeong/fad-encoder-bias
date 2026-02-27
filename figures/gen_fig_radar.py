#!/usr/bin/env python3
"""
Figure 1: Radar chart — four-axis encoder trade-off.

Usage:  python figures/gen_fig_radar.py [--input path/to/fad_results.json]
Output: figures/output/fig_radar.pdf + fig_radar.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RESULTS_DIR, FIGURES_DIR, ENCODER_KEYS, DATASETS,
    RECALL_LABELS, PRECISION_LABELS, SEMANTIC_LABELS, STRUCTURAL_LABELS,
)
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


def compute_scores(data):
    fad_max = {}
    for enc in ENCODER_KEYS:
        vals = [d["fad"] for d in data if d["encoder"] == enc]
        fad_max[enc] = max(vals)

    scores = {}
    for enc in ENCODER_KEYS:
        all_rec, all_prec, all_sem, all_stru = [], [], [], []
        for ds in DATASETS:
            all_rec  += [s_norm(get_fad(data, enc, l, ds), fad_max[enc])
                         for l in RECALL_LABELS if get_fad(data, enc, l, ds) is not None]
            all_prec += [s_norm(get_fad(data, enc, l, ds), fad_max[enc])
                         for l in PRECISION_LABELS if get_fad(data, enc, l, ds) is not None]
            all_sem  += [s_norm(get_fad(data, enc, l, ds), fad_max[enc])
                         for l in SEMANTIC_LABELS if get_fad(data, enc, l, ds) is not None]
            all_stru += [s_norm(get_fad(data, enc, l, ds), fad_max[enc])
                         for l in STRUCTURAL_LABELS if get_fad(data, enc, l, ds) is not None]
        # Order: Recall, Precision, Semantic, Structural
        scores[enc] = [
            1 - np.mean(all_rec),
            np.mean(all_prec),
            np.mean(all_sem),
            np.mean(all_stru),
        ]
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()
    input_path = Path(args.input) if args.input else RESULTS_DIR / "fad_results.json"
    data = load_data(input_path)
    scores = compute_scores(data)

    N = 4
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    AXIS_LABELS = [
        (0, "Recall",               "center", "bottom", 0.10),
        (1, "Precision",            "left",   "center", 0.14),
        (2, "Semantic\nAlignment",  "center", "top",    0.08),
        (3, "Structural\nAlignment", "right",  "center", 0.14),
    ]

    CFG = {
        "audiomae": ("#1f77b4", "-",  "o", 2.5, 15, "AudioMAE"),
        "encodec":  ("#ff7f0e", "-",  "v", 2.0, 17, "EnCodec"),
        "wav2vec":  ("#8c564b", "--", "*", 2.0, 20, "Wav2Vec 2.0"),
        "vggish":   ("#2ca02c", ":",  "^", 2.5, 17, "VGGish"),
        "clap":     ("#9467bd", "-.", "D", 2.0, 14, "CLAP"),
        "whisper":  ("#d62728", "--", "s", 2.5, 15, "Whisper"),
    }

    DRAW_ORDER   = ["clap", "wav2vec", "encodec", "vggish", "audiomae", "whisper"]
    LEGEND_ORDER = ["audiomae", "encodec", "wav2vec", "vggish", "clap", "whisper"]
    HIGHLIGHT    = ["audiomae", "whisper", "vggish"]

    fig, ax = plt.subplots(figsize=(9.5, 9.4), subplot_kw=dict(polar=True))
    ax.set_position([0.15, 0.22, 0.95, 0.90])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color="#999999", linewidth=0.6, linestyle="-", alpha=0.6)
    ax.xaxis.grid(True, color="#bbbbbb", linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    for idx, txt, ha, va, roff in AXIS_LABELS:
        ax.text(angles[idx], 1.08 + roff, txt,
                fontsize=25, fontweight="bold", color="#222222", ha=ha, va=va)

    for r in [0.25, 0.50, 0.75, 1.00]:
        ax.text(angles[0], r, f" {r:.2f}",
                fontsize=15, color="#555555", ha="left", va="bottom")

    for enc in DRAW_ORDER:
        vals = scores[enc] + [scores[enc][0]]
        c, ls, mk, lw, ms, lb = CFG[enc]
        zo = 5 if enc in HIGHLIGHT else 3
        ax.plot(angles, vals, color=c, linestyle=ls, linewidth=lw,
                marker=mk, markersize=ms, label=lb, zorder=zo,
                markeredgewidth=0.5, markeredgecolor="white")
        if enc in HIGHLIGHT:
            ax.fill(angles, vals, alpha=0.04, color=c, zorder=1)

    handles, labels = ax.get_legend_handles_labels()
    lbl_map = {l: h for h, l in zip(handles, labels)}
    oh = [lbl_map[CFG[e][5]] for e in LEGEND_ORDER]
    ol = [CFG[e][5] for e in LEGEND_ORDER]
    fig.legend(oh, ol, loc="lower center", ncol=6,
               fontsize=23, frameon=False, columnspacing=0.5,
               handletextpad=0.3, handlelength=1.5,
               bbox_to_anchor=(0.61, -0.01))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fig_radar.pdf",
                bbox_inches="tight", pad_inches=0.03, dpi=300)
    fig.savefig(FIGURES_DIR / "fig_radar.png",
                bbox_inches="tight", pad_inches=0.03, dpi=150)
    plt.close(fig)
    print("fig_radar.pdf / fig_radar.png saved")


if __name__ == "__main__":
    main()
