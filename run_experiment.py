#!/usr/bin/env python3
"""
Main experiment pipeline.

Pipeline per file:  Perturbation → Loudness Norm (-23 LUFS) → Resample → Encode
Then:               Compute μ, Σ of reference & perturbed → FAD

Usage:
    python run_experiment.py                           # Run all conditions
    python run_experiment.py --encoders whisper vggish  # Specific encoders
    python run_experiment.py --resume                   # Resume from saved results
"""

import argparse
import csv
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from config import (
    CACHE_DIR, RESULTS_DIR, TARGET_LUFS,
    LIBRI_DIR, ESC50_DIR, ENCODER_KEYS,
    build_conditions,
)
from preprocess import (
    load_audio, loudness_normalize,
    get_librispeech_files, get_esc50_files,
    download_librispeech, download_esc50,
)
from perturbations import apply_perturbation
from compute_fad import compute_statistics, frechet_distance
from encoders import ENCODER_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "experiment.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_JSON = RESULTS_DIR / "fad_results.json"
RESULTS_CSV = RESULTS_DIR / "fad_results.csv"


# ═══════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_embeddings(encoder, files, target_sr,
                       task=None, params=None, desc=""):
    """
    Extract one mean-pooled embedding per audio file.

    Returns: (N, D) numpy array.
    """
    all_embs = []
    for i, fp in enumerate(tqdm(files, desc=desc, leave=False, ncols=100)):
        try:
            wav, sr = load_audio(fp)

            if task is not None:
                wav = apply_perturbation(wav, sr, task, params)

            wav = loudness_normalize(wav, sr, TARGET_LUFS)

            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            emb = encoder.encode(wav, target_sr)
            if emb.ndim == 2:
                emb = emb.mean(axis=0)
            all_embs.append(emb)
        except Exception as e:
            if len(all_embs) == 0 and i < 5:
                logger.warning(f"  file error ({fp.name}): {e}")
            logger.debug(f"skip {fp.name}: {e}")
            continue

        if (i + 1) % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if not all_embs:
        raise RuntimeError(f"No embeddings extracted ({desc})")
    return np.stack(all_embs)


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def save_results(all_results):
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "encoder", "question", "task", "label", "dataset", "fad"])
        w.writeheader()
        w.writerows(all_results)


def load_existing_results():
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════

def run(encoder_keys=None, resume=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Ensure datasets exist
    from config import DATA_DIR
    download_librispeech(DATA_DIR)
    download_esc50(DATA_DIR)

    ls_files = get_librispeech_files(LIBRI_DIR)
    esc_files = get_esc50_files(ESC50_DIR)
    dataset_map = {
        "librispeech": ls_files,
        "esc50":       esc_files,
    }

    conditions = build_conditions()
    logger.info(f"Total conditions: {len(conditions)}")

    all_results = load_existing_results() if resume else []
    done_keys = {
        (r["encoder"], r["label"], r["dataset"])
        for r in all_results if np.isfinite(r.get("fad", float("nan")))
    }
    if done_keys:
        logger.info(f"[RESUME] {len(done_keys)} conditions already done")

    if encoder_keys is None:
        encoder_keys = ENCODER_KEYS

    for enc_key in encoder_keys:
        logger.info(f"\n{'='*60}")
        logger.info(f"  ENCODER: {enc_key}")
        logger.info(f"{'='*60}")

        try:
            encoder = ENCODER_REGISTRY[enc_key](device)
        except Exception as e:
            logger.error(f"Failed to load {enc_key}: {e}")
            continue

        tgt_sr = encoder.target_sr

        # Reference embeddings (original, unperturbed)
        ref_stats = {}
        for ds_name, files in dataset_map.items():
            cache_mu = CACHE_DIR / f"{enc_key}_ref_{ds_name}_mu.npy"
            cache_sig = CACHE_DIR / f"{enc_key}_ref_{ds_name}_sigma.npy"
            if cache_mu.exists() and cache_sig.exists():
                mu = np.load(cache_mu)
                sig = np.load(cache_sig)
                logger.info(f"  [cache] {ds_name} ref stats loaded")
            else:
                logger.info(f"  Computing reference embeddings: {ds_name}")
                embs = extract_embeddings(encoder, files, tgt_sr,
                                          desc=f"ref-{ds_name}")
                mu, sig = compute_statistics(embs)
                np.save(cache_mu, mu)
                np.save(cache_sig, sig)
                logger.info(f"  {ds_name} ref: emb shape {embs.shape}")
            ref_stats[ds_name] = (mu, sig)

        # Perturbed conditions
        for cond in conditions:
            for ds_name in cond["datasets"]:
                key = (enc_key, cond["label"], ds_name)
                if key in done_keys:
                    logger.info(f"  [SKIP] {cond['label']}|{ds_name}")
                    continue

                files = dataset_map[ds_name]
                mu_ref, sig_ref = ref_stats[ds_name]
                label = f"{cond['label']}|{ds_name}"
                logger.info(f"  → {label}")
                t0 = time.time()

                try:
                    embs = extract_embeddings(
                        encoder, files, tgt_sr,
                        task=cond["task"], params=cond["params"],
                        desc=label,
                    )
                    mu_e, sig_e = compute_statistics(embs)
                    fad = frechet_distance(mu_ref, sig_ref, mu_e, sig_e)
                except Exception as e:
                    logger.error(f"    ERROR: {e}")
                    fad = float("nan")

                elapsed = time.time() - t0
                logger.info(f"    FAD = {fad:.6f}  ({elapsed:.1f}s)")

                result = dict(
                    encoder=enc_key,
                    question=cond["q"],
                    task=cond["task"],
                    label=cond["label"],
                    dataset=ds_name,
                    fad=fad,
                )
                all_results.append(result)
                done_keys.add(key)
                save_results(all_results)

                gc.collect()
                torch.cuda.empty_cache()

        del encoder
        torch.cuda.empty_cache()
        logger.info(f"  {enc_key} done — GPU memory cleared")

    save_results(all_results)
    logger.info(f"\nExperiment complete. {len(all_results)} results saved.")
    logger.info(f"  JSON: {RESULTS_JSON}")
    logger.info(f"  CSV:  {RESULTS_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Run FAD encoder bias experiment")
    parser.add_argument("--encoders", nargs="+", default=None,
                        choices=ENCODER_KEYS,
                        help="Encoders to run (default: all)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (ignore previous results)")
    args = parser.parse_args()
    run(encoder_keys=args.encoders, resume=not args.no_resume)


if __name__ == "__main__":
    main()
