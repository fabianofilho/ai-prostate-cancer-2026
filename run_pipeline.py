# -*- coding: utf-8 -*-
"""
Main pipeline entrypoint.

This script orchestrates the full machine learning pipeline for prostate cancer
survival prediction. It can be run from the command line with optional arguments
to control which steps are executed.

Usage:
    # Run the full classification pipeline
    python run_pipeline.py

    # Run only specific steps
    python run_pipeline.py --steps preprocess select_features train evaluate

    # Run the survival analysis pipeline
    python run_pipeline.py --steps survival

    # Run everything
    python run_pipeline.py --steps all
"""

import argparse
import sys
from src.utils import create_directories, get_logger
from src import config

logger = get_logger("pipeline")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prostate Cancer Survival Prediction Pipeline"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["preprocess", "select_features", "train", "evaluate", "survival", "all"],
        default=["all"],
        help="Pipeline steps to execute. Default: all"
    )
    return parser.parse_args()


def main():
    """Orchestrates the full pipeline."""
    args = parse_args()
    steps = args.steps
    run_all = "all" in steps

    logger.info("=" * 60)
    logger.info("  Prostate Cancer Survival Prediction Pipeline")
    logger.info("  ASCO 2026 — FMUSP / LABDAPS / USP")
    logger.info("=" * 60)

    # Ensure all output directories exist
    create_directories(config.DIRS)

    # --- Step 1: Preprocessing ---
    if run_all or "preprocess" in steps:
        logger.info("\n>>> STEP 1: Preprocessing")
        from src.preprocess import main as preprocess_main
        preprocess_main()

    # --- Step 2: Feature Selection ---
    if run_all or "select_features" in steps:
        logger.info("\n>>> STEP 2: Feature Selection (Boruta)")
        from src.select_features import select_features
        select_features()

    # --- Step 3: Model Training ---
    if run_all or "train" in steps:
        logger.info("\n>>> STEP 3: Model Training and Hyperparameter Tuning")
        from src.train import train_models
        train_models()

    # --- Step 4: Model Evaluation ---
    if run_all or "evaluate" in steps:
        logger.info("\n>>> STEP 4: Model Evaluation and SHAP Analysis")
        from src.evaluate import evaluate_models
        evaluate_models()

    # --- Step 5: Survival Analysis ---
    if run_all or "survival" in steps:
        logger.info("\n>>> STEP 5: Survival Analysis (sksurv + SHAP)")
        try:
            from src.survival_analysis import main as survival_main
            survival_main()
        except ImportError:
            logger.error(
                "scikit-survival is not installed. "
                "Run: pip install scikit-survival"
            )
            sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("  Pipeline completed successfully!")
    logger.info("  Results saved to: results/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
