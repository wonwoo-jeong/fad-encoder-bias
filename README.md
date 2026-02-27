# FAD Encoder Bias: Reproduction Code

> **Paper:** *An Empirical Analysis of Task-Induced Encoder Bias in Fréchet Audio Distance*
>
> Submitted to Interspeech 2026
>
> Wonwoo Jeong
>
> **Keywords:** audio evaluation, Fréchet Audio Distance, text-to-audio generation, audio encoders, evaluation metrics

This repository provides full reproduction code for all experiments, figures, and tables in the paper. Given the same datasets and pretrained model weights, running the pipeline will produce results identical to those reported.

---

## Repository Structure

```
fad-encoder-bias/
├── config.py              # Paths, constants, perturbation condition definitions
├── preprocess.py          # Dataset download, LUFS normalization, audio I/O
├── perturbations.py       # 8 DSP perturbation functions (R/P/S/A axes)
├── encoders.py            # 6 pretrained encoder wrappers
├── compute_fad.py         # FAD computation (Eq. 1) + S_norm normalization (Eq. 3)
├── run_experiment.py      # Main pipeline: perturb → normalize → embed → FAD
├── analyze.py             # Score computation (Table 2) + reference verification
├── figures/
│   ├── gen_fig_radar.py       # Figure 1: Radar chart
│   ├── gen_fig_fidelity.py    # Figure 2: Raw vs Normalized FAD (noise)
│   ├── gen_fig_diversity.py   # Figure 3: Pitch-shift trajectory
│   └── gen_fig_blindspots.py  # Figure 4: Structural vs Semantic butterfly
├── results/
│   └── fad_results_reference.json   # Reference results for verification
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**System requirements:**
- Python 3.9+
- CUDA-compatible GPU (recommended, 16 GB+ VRAM)
- ~10 GB disk space for datasets + model weights

### 2. Download Datasets

```bash
python preprocess.py
```

This downloads:
- **LibriSpeech test-clean** (2,620 samples, 16 kHz) — [OpenSLR](https://www.openslr.org/12)
- **ESC-50** (2,000 samples, 44.1 kHz) — [GitHub](https://github.com/karolpiczak/ESC-50)

### 3. Run Full Experiment

```bash
python run_experiment.py
```

The pipeline runs 6 encoders × ~40 conditions × 1–2 datasets. On a single A100 GPU, this takes approximately 8–12 hours. The script supports automatic resumption — if interrupted, re-run the same command to continue.

**Partial runs** (for testing):
```bash
python run_experiment.py --encoders whisper vggish
```

### 4. Analyze Results & Reproduce Table 2

```bash
python analyze.py
python analyze.py --verify   # Compare against reference results
```

### 5. Generate Figures

```bash
python figures/gen_fig_radar.py       # Figure 1
python figures/gen_fig_fidelity.py    # Figure 2
python figures/gen_fig_diversity.py   # Figure 3
python figures/gen_fig_blindspots.py  # Figure 4
```

All figures are saved to `figures/output/` as PDF and PNG.

---

## Pipeline Details

### Overview

We decompose FAD evaluation into **Recall**, **Precision**, and **Alignment** (semantic + structural), using **log-scale self-reference normalization** ($S_\text{norm}$) for fair cross-encoder comparison. Controlled experiments on six encoders across two datasets reveal a *four-axis trade-off*: no single encoder functions as a universal evaluator, underscoring the need for evaluation-native encoders intrinsically aligned with human perception.

### Preprocessing (§3.1)

For each audio file, the pipeline applies the following in order:

1. **Perturbation** at native sample rate
2. **Loudness normalization** to −23 LUFS (ITU-R BS.1770-4)
3. **Resampling** to the encoder's native sample rate (16k / 24k / 48k)

No padding or truncation is applied; original audio length is preserved.

### Perturbation Design (§3.3)

| Axis | Perturbation | Parameters |
|---|---|---|
| **Recall** | Pitch shift | ±1, ±2 st |
| | Time stretch | 0.9×, 1.1× |
| **Precision** | White noise | SNR 60, 40, 20, 10, 0, −5 dB |
| | Low-pass filter | Cutoff 8000, 6000, 4000, 2000, 1000 Hz |
| | Reverb | RT60 0.1–2.0 s |
| **Semantic Alignment** | Extreme pitch shift | ±4, ±8 st |
| | Formant shift | 1.3×, 1.4× (F0 preserved) |
| **Structural Alignment** | Time reversal | — |
| | Chunk shuffle | 100, 250, 500, 1000 ms (10 ms crossfade) |

### Encoders (§3.2)

| Encoder | Training Task | Dim | Sample Rate |
|---|---|---|---|
| AudioMAE | Masked Reconstruction | 768 | 16 kHz |
| EnCodec 24 kHz | Neural Audio Compression | 128 | 24 kHz |
| Wav2Vec 2.0 Base | Contrastive Learning (SSL) | 768 | 16 kHz |
| VGGish | Audio Classification | 128 | 16 kHz |
| CLAP (LAION-630k) | Cross-modal Contrastive Learning | 512 | 48 kHz |
| Whisper large-v3 | Automatic Speech Recognition | 1280 | 16 kHz |

For transformer-based encoders (Whisper, AudioMAE, Wav2Vec 2.0), the final hidden state is extracted; for EnCodec, the continuous encoder output prior to the residual vector quantizer is used. Frame-level outputs are temporal mean-pooled to produce one clip-level embedding per file; CLAP natively produces a clip-level embedding via internal attention pooling.

### FAD & Normalization (§2.2–2.3)

**FAD** (Eq. 1): Fréchet distance between Gaussian fits of reference and perturbed embeddings.

**S_norm** (Eq. 3): Log-scale self-reference normalization:

$$S_{\text{norm}} = \frac{\log(1 + \text{FAD})}{\log(1 + \text{FAD}_{\max})}$$

where FAD_max is the maximum FAD observed for a given encoder across all perturbations and both datasets.

---

## Verification

The repository includes `results/fad_results_reference.json` containing the exact FAD values reported in the paper. After running the experiment:

```bash
python analyze.py --verify
```

This compares every individual FAD value and all Table 2 scores against the reference.

> **Note on minor numerical differences:**
> A small subset of perturbations involves stochastic operations (e.g., White Noise injects `torch.randn`-generated noise; Formant Shift relies on Parselmouth's internal DSP). Due to floating-point non-determinism across hardware and library versions, re-computed FAD values for these conditions may differ by < 1 % on average. All **Table 2 normalized scores** and derived statistics (Pearson *r*, axis rankings) are robust to these variations and will match the paper exactly.

---

## Citation

```bibtex
@article{jeong2025fad,
  title={An Empirical Analysis of Task-Induced Encoder Bias in {Fr\'{e}chet} Audio Distance},
  author={Jeong, Wonwoo},
  journal={arXiv preprint arXiv:2602.XXXXX},
  year={2026}
}
```

## License

This code is released for research purposes. The pretrained model weights are subject to their respective licenses (OpenAI Whisper, Meta EnCodec/Wav2Vec 2.0, LAION CLAP, Google VGGish, AudioMAE).
