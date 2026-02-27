#!/usr/bin/env python3
"""
Comprehensive verification: check that all numerical claims in the paper
are consistent with fad_results.json.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from config import RESULTS_DIR, ENCODER_KEYS, DATASETS
from compute_fad import s_norm

results_path = RESULTS_DIR / "fad_results.json"
with open(results_path) as f:
    data = json.load(f)

lookup = {(d["encoder"], d["label"], d["dataset"]): d["fad"] for d in data}

fad_max = {}
for enc in ENCODER_KEYS:
    vals = [d["fad"] for d in data if d["encoder"] == enc]
    fad_max[enc] = max(vals)

# ── Table 2 (paper values vs computed) ──
from analyze import compute_table2
table = compute_table2(data)

PAPER_TABLE2 = {
    "audiomae": {"recall": 0.645, "precision": 0.463, "semantic": 0.300, "structural": 0.238},
    "encodec":  {"recall": 0.851, "precision": 0.450, "semantic": 0.254, "structural": 0.042},
    "wav2vec":  {"recall": 0.767, "precision": 0.420, "semantic": 0.294, "structural": 0.170},
    "vggish":   {"recall": 0.580, "precision": 0.380, "semantic": 0.445, "structural": 0.140},
    "clap":     {"recall": 0.694, "precision": 0.309, "semantic": 0.261, "structural": 0.238},
    "whisper":  {"recall": 0.889, "precision": 0.147, "semantic": 0.119, "structural": 0.495},
}

errors = []

print("=" * 70)
print("  TABLE 2 VERIFICATION")
print("=" * 70)
for enc in ENCODER_KEYS:
    for axis in ["recall", "precision", "semantic", "structural"]:
        computed = round(table[enc][axis], 3)
        paper = PAPER_TABLE2[enc][axis]
        ok = abs(computed - paper) < 0.002
        status = "OK" if ok else "FAIL"
        if not ok:
            errors.append(f"Table2 {enc}/{axis}: paper={paper}, computed={computed}")
        print(f"  [{status}] {enc:10s} {axis:12s}: paper={paper:.3f}, computed={computed:.3f}")

# ── Pearson r (Structural vs Semantic) ──
struct_vals = [table[e]["structural"] for e in ENCODER_KEYS]
sem_vals = [table[e]["semantic"] for e in ENCODER_KEYS]
r = np.corrcoef(struct_vals, sem_vals)[0, 1]
paper_r = -0.67
ok = abs(r - paper_r) < 0.02
print(f"\n  [{'OK' if ok else 'FAIL'}] Pearson r: paper={paper_r:.2f}, computed={r:.2f}")
if not ok:
    errors.append(f"Pearson r: paper={paper_r}, computed={r:.2f}")

# ── FAD_max values ──
print("\n" + "=" * 70)
print("  FAD_MAX VERIFICATION")
print("=" * 70)
for enc in ENCODER_KEYS:
    print(f"  {enc:10s}: {fad_max[enc]:.4f}")

# ── Summary ──
print("\n" + "=" * 70)
if errors:
    print(f"  FAILED: {len(errors)} mismatches")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ALL PAPER CLAIMS VERIFIED SUCCESSFULLY")
    sys.exit(0)
