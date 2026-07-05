"""
drift_detection.py
------------------
Data drift detection for the heart disease MLOps pipeline.

Compares incoming (production) data against the training reference using:
1. Kolmogorov-Smirnov (KS) test  -> distribution shift per feature
2. Population Stability Index (PSI) -> magnitude of shift per feature

Outputs a JSON drift report and an overall drift flag that can gate
retraining in CI/CD or scheduled jobs.

Usage:
    python -m monitoring.drift_detection                      # test.csv vs train.csv
    python -m monitoring.drift_detection --new path/to.csv    # custom batch
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils import TRAIN_PATH, TEST_PATH, TARGET_COL, REPORTS_DIR

KS_PVALUE_THRESHOLD = 0.05      # drift if p < 0.05
PSI_THRESHOLD = 0.2             # >0.2 = significant shift, 0.1-0.2 = moderate
DRIFT_FEATURE_RATIO = 0.3       # overall drift if >=30% of features drifted


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D samples."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Bin edges from the reference distribution (quantiles are robust)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_feature_drift(reference: pd.Series, current: pd.Series) -> dict:
    """Run KS test + PSI for one feature."""
    ks_stat, p_value = ks_2samp(reference.dropna(), current.dropna())
    psi = compute_psi(reference.dropna().values, current.dropna().values)

    drifted = bool(p_value < KS_PVALUE_THRESHOLD or psi > PSI_THRESHOLD)

    return {
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(p_value), 4),
        "psi": round(psi, 4),
        "drift_detected": drifted,
    }


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Full drift report across all shared numeric features."""
    features = [
        c for c in reference_df.columns
        if c != TARGET_COL and c in current_df.columns
        and pd.api.types.is_numeric_dtype(reference_df[c])
    ]

    results = {f: detect_feature_drift(reference_df[f], current_df[f]) for f in features}

    n_drifted = sum(r["drift_detected"] for r in results.values())
    overall = n_drifted / max(len(features), 1) >= DRIFT_FEATURE_RATIO

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_features_checked": len(features),
        "n_features_drifted": n_drifted,
        "drifted_features": [f for f, r in results.items() if r["drift_detected"]],
        "overall_drift_detected": overall,
        "retraining_recommended": overall,
        "thresholds": {
            "ks_p_value": KS_PVALUE_THRESHOLD,
            "psi": PSI_THRESHOLD,
            "feature_ratio": DRIFT_FEATURE_RATIO,
        },
        "features": results,
    }


def save_report(report: dict, output_dir: str = None) -> str:
    output_dir = output_dir or REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "drift_report.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Detect data drift.")
    parser.add_argument("--reference", default=TRAIN_PATH, help="Reference (training) CSV")
    parser.add_argument("--new", default=TEST_PATH, help="New/production data CSV")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.new)

    report = detect_drift(reference, current)
    path = save_report(report)

    print(f"Drift report saved to {path}")
    print(f"Features drifted: {report['n_features_drifted']}/{report['n_features_checked']}")
    if report["overall_drift_detected"]:
        print("DRIFT DETECTED - retraining recommended")
    else:
        print("No significant drift detected")


if __name__ == "__main__":
    main()
