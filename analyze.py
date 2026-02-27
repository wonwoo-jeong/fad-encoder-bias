#!/usr/bin/env python3
"""
Result analysis and verification.

Computes Table 2 axis scores, Pearson correlations, and optionally
verifies against a reference results file.

Usage:
    python analyze.py                           # Analyze results/fad_results.json
    python analyze.py --verify                  # Also verify against reference
    python analyze.py --input path/to/results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from config import (
    RESULTS_DIR, ENCODER_KEYS, DATASETS,
    RECALL_LABELS, PRECISION_LABELS,
    SEMANTIC_LABELS, STRUCTURAL_LABELS,
)
from compute_fad import s_norm


def load_results(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def build_lookup(results: list) -> dict:
    """(encoder, label, dataset) -> fad"""
    return {(d["encoder"], d["label"], d["dataset"]): d["fad"] for d in results}


def compute_fad_max_all(results: list) -> dict:
    """FAD_max per encoder across ALL conditions and datasets."""
    fad_max = {}
    for enc in ENCODER_KEYS:
        vals = [d["fad"] for d in results if d["encoder"] == enc]
        fad_max[enc] = max(vals) if vals else 0
    return fad_max


def compute_axis_score(enc: str, labels: list, lookup: dict,
                       fad_max: dict, datasets=None) -> float:
    """Average S_norm for an encoder across given labels and datasets."""
    if datasets is None:
        datasets = DATASETS
    vals = []
    for label in labels:
        for ds in datasets:
            k = (enc, label, ds)
            if k in lookup:
                vals.append(s_norm(lookup[k], fad_max[enc]))
    return np.mean(vals) if vals else float("nan")


def compute_table2(results: list) -> dict:
    """Compute Table 2: Rec. | Prec. | Sem. | Struct. for each encoder."""
    lookup = build_lookup(results)
    fad_max = compute_fad_max_all(results)
    table = {}
    for enc in ENCODER_KEYS:
        recall_raw = compute_axis_score(enc, RECALL_LABELS, lookup, fad_max)
        table[enc] = {
            "recall": 1 - recall_raw,
            "precision": compute_axis_score(enc, PRECISION_LABELS, lookup, fad_max),
            "semantic": compute_axis_score(enc, SEMANTIC_LABELS, lookup, fad_max),
            "structural": compute_axis_score(enc, STRUCTURAL_LABELS, lookup, fad_max),
        }
    return table


def print_table2(table: dict):
    print("\n" + "=" * 62)
    print("  TABLE 2: Normalized Axis Scores (Rec. | Prec. | Sem. | Struct.)")
    print("=" * 62)
    print(f"  {'Encoder':<12s} {'Recall':>8s} {'Prec.':>8s} {'Sem.':>8s} {'Struct.':>8s}")
    print("  " + "-" * 48)
    for enc in ENCODER_KEYS:
        s = table[enc]
        print(f"  {enc:<12s} {s['recall']:8.3f} {s['precision']:8.3f} "
              f"{s['semantic']:8.3f} {s['structural']:8.3f}")

    # Best per column
    for axis in ["recall", "precision", "semantic", "structural"]:
        best = max(ENCODER_KEYS, key=lambda e: table[e][axis])
        print(f"  Best {axis}: {best} ({table[best][axis]:.3f})")

    # Pearson correlation: Structural vs Semantic
    struct_vals = [table[e]["structural"] for e in ENCODER_KEYS]
    sem_vals = [table[e]["semantic"] for e in ENCODER_KEYS]
    r = np.corrcoef(struct_vals, sem_vals)[0, 1]
    print(f"\n  Pearson r (Structural vs Semantic): {r:.2f}")


def verify_against_reference(results: list, ref_path: Path,
                             tol: float = 1e-3) -> bool:
    """Compare computed scores against reference results."""
    ref_results = load_results(ref_path)
    ref_lookup = build_lookup(ref_results)
    my_lookup = build_lookup(results)

    ref_table = compute_table2(ref_results)
    my_table = compute_table2(results)

    all_ok = True

    print("\n" + "=" * 62)
    print("  VERIFICATION against reference")
    print("=" * 62)

    # Compare Table 2 scores
    for enc in ENCODER_KEYS:
        for axis in ["recall", "precision", "semantic", "structural"]:
            ref_val = ref_table[enc][axis]
            my_val = my_table[enc][axis]
            diff = abs(ref_val - my_val)
            status = "OK" if diff < tol else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
                print(f"  [{status}] {enc} {axis}: ref={ref_val:.4f}, "
                      f"got={my_val:.4f}, diff={diff:.6f}")

    # Compare individual FAD values
    mismatches = 0
    for key, ref_fad in ref_lookup.items():
        if key in my_lookup:
            diff = abs(ref_fad - my_lookup[key])
            if diff > tol * max(1.0, ref_fad):
                mismatches += 1
                if mismatches <= 5:
                    print(f"  [FAD MISMATCH] {key}: ref={ref_fad:.4f}, "
                          f"got={my_lookup[key]:.4f}")

    if mismatches > 5:
        print(f"  ... and {mismatches - 5} more FAD mismatches")

    if all_ok and mismatches == 0:
        print("  All values match reference within tolerance.")
    else:
        print(f"\n  Table score mismatches: {'none' if all_ok else 'FOUND'}")
        print(f"  FAD value mismatches: {mismatches}")

    return all_ok and mismatches == 0


def main():
    parser = argparse.ArgumentParser(description="Analyze FAD experiment results")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to fad_results.json")
    parser.add_argument("--verify", action="store_true",
                        help="Verify against reference results")
    parser.add_argument("--ref", type=str, default=None,
                        help="Path to reference results")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else RESULTS_DIR / "fad_results.json"
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run the experiment first.")
        sys.exit(1)

    results = load_results(input_path)
    print(f"Loaded {len(results)} results from {input_path}")

    # FAD_max per encoder
    fad_max = compute_fad_max_all(results)
    print("\nFAD_max per encoder:")
    for enc in ENCODER_KEYS:
        print(f"  {enc}: {fad_max[enc]:.4f}")

    table = compute_table2(results)
    print_table2(table)

    if args.verify:
        ref_path = Path(args.ref) if args.ref else RESULTS_DIR / "fad_results_reference.json"
        if not ref_path.exists():
            print(f"Reference file not found: {ref_path}")
            sys.exit(1)
        ok = verify_against_reference(results, ref_path)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
