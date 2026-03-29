# -*- coding: utf-8 -*-
"""
Feature selection script using the Boruta algorithm.

This script loads the processed data, splits it into training and testing sets,
and applies the Boruta feature selection algorithm to identify the most
relevant predictors for the model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
import joblib

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

def select_features():
    """Performs feature selection using Boruta."""
    logger.info("Starting feature selection process...")

    # 1. Load processed data
    try:
        df = pd.read_parquet(config.PROCESSED_DATA_FILE)
        logger.info(f"Loaded processed data from {config.PROCESSED_DATA_FILE}")
    except FileNotFoundError:
        logger.error(f"Processed data file not found. Run preprocess.py first.")
        raise

    # 2. Split data
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        stratify=y, 
        random_state=config.RANDOM_STATE
    )
    logger.info("Data split into training and testing sets.")

    # 3. Configure and run Boruta
    logger.info("Configuring and running BorutaPy...")
    rf_boruta = RandomForestClassifier(n_jobs=-1, class_weight="balanced", **config.BORUTA_PARAMS)
    feat_selector = BorutaPy(rf_boruta, n_estimators="auto", verbose=0, random_state=config.RANDOM_STATE)
    
    feat_selector.fit(X_train.values, y_train.values)

    # 4. Get selected features and handle fallback
    cols_selected = X_train.columns[feat_selector.support_].tolist()

    if len(cols_selected) < 2:
        logger.warning("Boruta was too restrictive. Falling back to top features from RandomForest.")
        rf_simple = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE).fit(X_train, y_train)
        cols_selected = pd.Series(rf_simple.feature_importances_, index=X_train.columns).nlargest(config.FALLBACK_N_FEATURES).index.tolist()

    logger.info(f"Selected {len(cols_selected)} features: {cols_selected}")

    # 5. Save results
    # Save the list of selected features
    features_path = config.DIRS["importances"] + "/selected_features.joblib"
    joblib.dump(cols_selected, features_path)
    logger.info(f"List of selected features saved to {features_path}")

    # Save the datasets with selected features
    train_df = pd.concat([X_train[cols_selected], y_train], axis=1)
    test_df = pd.concat([X_test[cols_selected], y_test], axis=1)

    train_df.to_parquet(config.TRAIN_DATA_FILE, index=False)
    test_df.to_parquet(config.TEST_DATA_FILE, index=False)
    logger.info(f"Train and test sets with selected features saved.")

if __name__ == "__main__":
    select_features()
