# -*- coding: utf-8 -*-
"""
Model training and hyperparameter tuning script.

This script loads the feature-selected training data, tunes hyperparameters for
multiple models using RandomizedSearchCV, trains them, and saves the best
estimators and performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import joblib

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

def train_models():
    """Tunes and trains multiple models."""
    logger.info("Starting model training and tuning process...")

    # 1. Load training data
    try:
        train_df = pd.read_parquet(config.TRAIN_DATA_FILE)
        logger.info(f"Loaded training data from {config.TRAIN_DATA_FILE}")
    except FileNotFoundError:
        logger.error("Training data not found. Run select_features.py first.")
        raise

    X_train = train_df.drop(columns=[config.TARGET_COL])
    y_train = train_df[config.TARGET_COL]

    # 2. Get models and hyperparameter grids
    models_map = config.MODELS_MAP
    param_grids = config.get_param_grids(y_train)

    trained_models = {}
    results_data = []

    # 3. Iterate through models, tune, and train
    for name, model in models_map.items():
        logger.info(f"--> Processing model: {name}")

        # Special handling for TabPFN (no hyperparameter tuning)
        if name == "TabPFN":
            # TabPFN has a limit of ~2k samples for fast inference
            if len(X_train) > 2000:
                idx = np.random.choice(len(X_train), 2000, replace=False)
                X_t, y_t = X_train.iloc[idx], y_train.iloc[idx]
            else:
                X_t, y_t = X_train, y_train
            
            model.fit(X_t, y_t)
            best_estimator = model
        else:
            # Randomized Search for other models
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[name],
                n_iter=config.N_ITER_RANDOM_SEARCH,
                scoring="f1",
                cv=config.CV_FOLDS,
                verbose=0,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
            rs.fit(X_train, y_train)
            best_estimator = rs.best_estimator_
            logger.info(f"    Best params for {name}: {rs.best_params_}")

        # Store the best estimator
        trained_models[name] = best_estimator
        model_path = config.DIRS["models"] + f"/{name}_best_model.pkl"
        joblib.dump(best_estimator, model_path)
        logger.info(f"    Saved best {name} model to {model_path}")

    logger.info("All models have been trained and saved.")

if __name__ == "__main__":
    train_models()
