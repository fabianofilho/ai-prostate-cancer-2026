# -*- coding: utf-8 -*-
"""
Centralized configuration file for the Prostate Cancer Survival Prediction project.

This file contains all the constants, paths, column lists, and model parameters
to ensure that the scripts are clean and easy to maintain.
"""

import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier

# --- 1. DIRECTORY AND PATHS CONFIGURATION ---
# Base path of the project. Assumes the script is run from the root directory.
BASE_PATH = os.getcwd()

# Dictionary of all relevant directories
DIRS = {
    "data_raw": os.path.join(BASE_PATH, "data"),
    "data_processed": os.path.join(BASE_PATH, "data", "processed"),
    "models": os.path.join(BASE_PATH, "results", "models"),
    "plots": os.path.join(BASE_PATH, "results", "plots"),
    "metrics": os.path.join(BASE_PATH, "results", "metrics"),
    "importances": os.path.join(BASE_PATH, "results", "importances")
}

# Input data file
RAW_DATA_FILE = os.path.join(DIRS["data_raw"], "RHC_SIM_ES_PROSTATA.csv")
PROCESSED_DATA_FILE = os.path.join(DIRS["data_processed"], "processed_data.parquet")
TRAIN_DATA_FILE = os.path.join(DIRS["data_processed"], "train.parquet")
TEST_DATA_FILE = os.path.join(DIRS["data_processed"], "test.parquet")


# --- 2. DATA PREPROCESSING AND FEATURE ENGINEERING ---

# Target variable and status mapping
TARGET_COL = "target"
STATUS_COL = "STATUS"
STATUS_MAP = {"ALIVE": 0, "PCA_DEATH": 1, "OTHER_DEATH": 1}

# Columns to be converted to specific types
COLS_NUMERIC = ["IDADE", "TEMPO_DIAS"]
COLS_DATE = ["DTNASC", "DTDIAG", "DTINIC", "DTULTINFO"]

# Threshold for dropping columns with low variance
LOW_VARIANCE_THRESHOLD = 0.90

# Columns to be removed after feature engineering or due to leakage
COLS_LEAKAGE_AND_REDUNDANT = [
    "STATUS", "target", "OBITO_CAN_PROSTATA", "DTOBITO", "TEMPO_DIAS", "TEMPO_ANO",
    "DTULTINFO", "RACACOR", "INSTRUC", "ESTCONJ", "PRITRATH", "ORIGENCA",
    "STATUS_OTHER_DEATH", "STATUS_PCA_DEATH", "DTOBITO/CENSURA", "ESTDFIMT",
    "DIAGANT", "BASMAIMP", "TIPOHIST_CAT", "TIPOHIST", "CAUBAMOR"
]


# --- 3. FEATURE SELECTION CONFIGURATION ---

# BorutaPy feature selector configuration
BORUTA_PARAMS = {
    "n_estimators": "auto",
    "verbose": 0,
    "random_state": 42,
    "max_depth": 7  # A bit deeper for more interaction checks
}

# Fallback: number of features to select if Boruta is too restrictive
FALLBACK_N_FEATURES = 15


# --- 4. MODEL TRAINING AND HYPERPARAMETER TUNING ---

# General training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER_RANDOM_SEARCH = 20

# Models to be trained
MODELS_MAP = {
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    "TabPFN": TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
}

# Hyperparameter grids for RandomizedSearchCV
def get_param_grids(y_train):
    try:
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    except ZeroDivisionError:
        ratio = 1.0

    return {
        "RandomForest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "class_weight": ["balanced", "balanced_subsample"]
        },
        "XGBoost": {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": [ratio, ratio * 1.2]
        },
        "LightGBM": {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 50, 100],
            "max_depth": [-1, 10, 20],
            "scale_pos_weight": [ratio]
        }
    }


# --- 5. EVALUATION AND PLOTTING ---

# Colors for comparison plots
PLOT_COLORS = {
    "RandomForest": "green",
    "XGBoost": "blue",
    "LightGBM": "purple",
    "TabPFN": "red"
}

# DPI for saving figures
FIGURE_DPI = 300


# --- 6. SURVIVAL ANALYSIS CONFIGURATION ---

from sksurv.ensemble import (
    RandomSurvivalForest,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

SURVIVAL_MODELS_MAP = {
    "RandomSurvivalForest": RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "ExtraSurvivalTrees": ExtraSurvivalTrees(
        n_estimators=200,
        min_samples_split=6,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "GradientBoostingSurvival": GradientBoostingSurvivalAnalysis(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        subsample=0.8,
        random_state=RANDOM_STATE,
    ),
    "CoxnetSurvival": CoxnetSurvivalAnalysis(
        l1_ratio=0.5,
        alpha_min_ratio=0.1,
        max_iter=1000,
        fit_baseline_model=True,
    ),
    "CoxPH": CoxPHSurvivalAnalysis(alpha=0.1),
}

# Models that require StandardScaler preprocessing before fit
SURVIVAL_MODELS_NEED_SCALING = {"CoxPH", "CoxnetSurvival"}

# Colors for survival model comparison plots
SURVIVAL_PLOT_COLORS = {
    "RandomSurvivalForest":     "steelblue",
    "ExtraSurvivalTrees":       "seagreen",
    "GradientBoostingSurvival": "darkorange",
    "CoxnetSurvival":           "purple",
    "CoxPH":                    "crimson",
}
