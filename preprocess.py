#!/usr/bin/env python3
"""
Data acquisition and audio preprocessing.

Usage:
    python preprocess.py            # Download both datasets
    python preprocess.py --check    # Verify datasets exist
"""

import argparse
import logging
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dataset download
# ═══════════════════════════════════════════════════════════════════════════

LIBRI_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def download_librispeech(data_dir: Path) -> Path:
    """Download and extract LibriSpeech test-clean."""
    target = data_dir / "LibriSpeech" / "test-clean"
    if target.exists() and any(target.rglob("*.flac")):
        logger.info(f"LibriSpeech test-clean already exists at {target}")
        return target

    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "test-clean.tar.gz"

    if not archive.exists():
        logger.info(f"Downloading LibriSpeech test-clean ({LIBRI_URL}) ...")
        urllib.request.urlretrieve(LIBRI_URL, str(archive))

    logger.info("Extracting LibriSpeech ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)

    n_files = len(list(target.rglob("*.flac")))
    logger.info(f"LibriSpeech test-clean: {n_files} files")
    return target


def download_esc50(data_dir: Path) -> Path:
    """Download and extract ESC-50."""
    target = data_dir / "ESC-50-master"
    if target.exists() and any((target / "audio").glob("*.wav")):
        logger.info(f"ESC-50 already exists at {target}")
        return target

    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ESC-50-master.zip"

    if not archive.exists():
        logger.info(f"Downloading ESC-50 ({ESC50_URL}) ...")
        urllib.request.urlretrieve(ESC50_URL, str(archive))

    logger.info("Extracting ESC-50 ...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(data_dir)

    n_files = len(list((target / "audio").glob("*.wav")))
    logger.info(f"ESC-50: {n_files} files")
    return target


# ═══════════════════════════════════════════════════════════════════════════
# 2. File listing
# ═══════════════════════════════════════════════════════════════════════════

def get_librispeech_files(libri_dir: Path) -> list:
    files = sorted(libri_dir.rglob("*.flac"))
    logger.info(f"LibriSpeech test-clean: {len(files)} files")
    return files


def get_esc50_files(esc50_dir: Path) -> list:
    files = sorted((esc50_dir / "audio").glob("*.wav"))
    logger.info(f"ESC-50: {len(files)} files")
    return files


# ═══════════════════════════════════════════════════════════════════════════
# 3. Audio I/O and normalization
# ═══════════════════════════════════════════════════════════════════════════

def load_audio(path) -> tuple:
    """Load mono audio at native sample rate. Returns (wav, sr)."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def loudness_normalize(wav: torch.Tensor, sr: int,
                       target_lufs: float = -23.0) -> torch.Tensor:
    """ITU-R BS.1770-4 loudness normalization via pyloudnorm."""
    import pyloudnorm as pyln

    audio = wav.squeeze().numpy().astype(np.float64)
    if audio.size == 0:
        return wav
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness):
        return wav
    try:
        normed = pyln.normalize.loudness(audio, loudness, target_lufs)
    except Exception:
        return wav
    return torch.from_numpy(normed).unsqueeze(0).float()


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--check", action="store_true",
                        help="Only check if datasets exist")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    from config import DATA_DIR
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    if args.check:
        libri = data_dir / "LibriSpeech" / "test-clean"
        esc50 = data_dir / "ESC-50-master" / "audio"
        ok = True
        if not libri.exists() or not any(libri.rglob("*.flac")):
            logger.error(f"LibriSpeech not found at {libri}")
            ok = False
        else:
            n = len(list(libri.rglob("*.flac")))
            logger.info(f"LibriSpeech: {n} files")

        if not esc50.exists() or not any(esc50.glob("*.wav")):
            logger.error(f"ESC-50 not found at {esc50}")
            ok = False
        else:
            n = len(list(esc50.glob("*.wav")))
            logger.info(f"ESC-50: {n} files")

        sys.exit(0 if ok else 1)

    download_librispeech(data_dir)
    download_esc50(data_dir)
    logger.info("All datasets ready.")


if __name__ == "__main__":
    main()
