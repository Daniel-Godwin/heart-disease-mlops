"""
run.py
------
Pipeline orchestrator for the heart disease MLOps project.

Runs pipeline stages in order, with clear logging and fail-fast behavior.

Usage:
    python run.py                    # full pipeline
    python run.py --stage preprocess # single stage
    python run.py --stage train
    python run.py --stage drift
    python run.py --stage explain

Stages:
    ingest      Load raw data -> data/processed/heart_processed.csv
    preprocess  Clean, encode, scale, split -> train.csv / test.csv (+ scaler)
    train       Train + tune model with MLflow tracking -> models/model.pkl
    drift       Data drift report -> reports/drift_report.json
    explain     SHAP global/local explanations
"""

import argparse
import sys
import time


def stage_ingest():
    from src import data_ingestion
    data_ingestion.main()


def stage_preprocess():
    from src import preprocessing
    preprocessing.main()


def stage_train():
    from src import train_model
    train_model.main()


def stage_drift():
    import pandas as pd
    from monitoring.drift_detection import detect_drift, save_report
    from src.utils import TRAIN_PATH, TEST_PATH

    reference = pd.read_csv(TRAIN_PATH)
    current = pd.read_csv(TEST_PATH)
    report = detect_drift(reference, current)
    path = save_report(report)
    print(f"Drift report saved to {path}")
    print(f"Features drifted: {report['n_features_drifted']}/{report['n_features_checked']}")
    if report["overall_drift_detected"]:
        print("⚠️ DRIFT DETECTED - retraining recommended")


def stage_explain():
    from monitoring import explainability
    if hasattr(explainability, "main"):
        explainability.main()


STAGES = {
    "ingest": stage_ingest,
    "preprocess": stage_preprocess,
    "train": stage_train,
    "drift": stage_drift,
    "explain": stage_explain,
}

# Full pipeline order (ingest is skipped by default because processed data
# is versioned in the repo; add --with-ingest to include it)
DEFAULT_ORDER = ["preprocess", "train", "drift"]


def run_stage(name: str) -> bool:
    print(f"\n{'=' * 60}\n▶ STAGE: {name.upper()}\n{'=' * 60}")
    start = time.time()
    try:
        STAGES[name]()
        print(f"✅ Stage '{name}' completed in {time.time() - start:.1f}s")
        return True
    except Exception as exc:
        print(f"❌ Stage '{name}' failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Heart disease MLOps pipeline runner")
    parser.add_argument("--stage", choices=list(STAGES), help="Run a single stage")
    parser.add_argument("--with-ingest", action="store_true",
                        help="Include raw-data ingestion in the full pipeline")
    args = parser.parse_args()

    if args.stage:
        order = [args.stage]
    else:
        order = (["ingest"] if args.with_ingest else []) + DEFAULT_ORDER

    print(f"Pipeline stages: {' -> '.join(order)}")

    for stage in order:
        if not run_stage(stage):
            print("\n🛑 Pipeline stopped due to stage failure.")
            sys.exit(1)

    print("\n🎉 Pipeline finished successfully.")


if __name__ == "__main__":
    main()
