# -*- coding: utf-8 -*-
"""
Model evaluation script.

This script loads the trained models and the test set to:
- Evaluate performance (F1-score, classification report, confusion matrix).
- Generate and save comparison plots (ROC, Precision-Recall).
- Compute and save SHAP values for the best model for interpretability.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

def evaluate_models():
    """Evaluates all trained models on the test set."""
    logger.info("Starting model evaluation process...")

    # 1. Load test data
    try:
        test_df = pd.read_parquet(config.TEST_DATA_FILE)
        X_test = test_df.drop(columns=[config.TARGET_COL])
        y_test = test_df[config.TARGET_COL]
        logger.info(f"Loaded test data from {config.TEST_DATA_FILE}")
    except FileNotFoundError:
        logger.error("Test data not found. Run select_features.py first.")
        raise

    # 2. Evaluate each model
    results_data = []
    trained_models = {}
    for name in config.MODELS_MAP.keys():
        model_path = f"{config.DIRS[\"models\"]}/{name}_best_model.pkl"
        try:
            model = joblib.load(model_path)
            trained_models[name] = model
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            results_data.append({"Model": name, "Test_F1": f1})
            logger.info(f"  -> Evaluated {name}: Test F1 = {f1:.4f}")
        except FileNotFoundError:
            logger.warning(f"Model file not found for {name} at {model_path}. Skipping.")

    # 3. Save model ranking
    if not results_data:
        logger.error("No models were evaluated. Exiting.")
        return

    ranking_df = pd.DataFrame(results_data).sort_values(by="Test_F1", ascending=False)
    ranking_df.to_csv(f"{config.DIRS[\"metrics\"]}/model_ranking_optimized.csv", index=False)
    logger.info(f"Model ranking saved to {config.DIRS[\"metrics\"]}/model_ranking_optimized.csv")

    best_model_name = ranking_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    logger.info(f"Best performing model: {best_model_name}")

    # 4. Generate plots and detailed reports for the best model
    generate_comparison_plots(trained_models, X_test, y_test)
    generate_final_report(best_model, best_model_name, X_test, y_test)
    generate_shap_summary(best_model, best_model_name, X_test)

def generate_comparison_plots(models, X_test, y_test):
    """Generates and saves ROC and PR comparison plots."""
    logger.info("Generating comparison plots (ROC and PR)..." )
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})", color=config.PLOT_COLORS.get(name, "black"))
    
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{config.DIRS[\"plots\"]}/all_models_roc_comparison.png", dpi=config.FIGURE_DPI)
    plt.close()

def generate_final_report(model, model_name, X_test, y_test):
    """Generates and saves the final classification report and confusion matrix."""
    logger.info(f"Generating final report for the best model: {model_name}")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Vivo", "Óbito"])
    with open(f"{config.DIRS[\"metrics\"]}/classification_report_{model_name}.txt", "w") as f:
        f.write(report)

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Pred: Vivo", "Pred: Óbito"], yticklabels=["Real: Vivo", "Real: Óbito"])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(f"{config.DIRS[\"plots\"]}/confusion_matrix_{model_name}.png", dpi=config.FIGURE_DPI)
    plt.close()

def generate_shap_summary(model, model_name, X_test):
    """Generates and saves SHAP summary plots."""
    logger.info(f"Generating SHAP summary for {model_name}...")
    try:
        if model_name in ["XGBoost", "RandomForest", "LightGBM"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        else: # Fallback for models like TabPFN
            logger.warning("Using KernelExplainer for SHAP, which can be slow.")
            sample_train = shap.sample(X_test, 50) # Use test data for background
            explainer = shap.KernelExplainer(model.predict_proba, sample_train)
            shap_values = explainer.shap_values(X_test.iloc[:50], nsamples=100)
            vals = shap_values[1]

        # SHAP Summary Plot (Beeswarm)
        plt.figure()
        shap.summary_plot(vals, X_test, show=False)
        plt.savefig(f"{config.DIRS[\"plots\"]}/shap_summary_beeswarm.png", dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close()

        # SHAP Importance Plot (Bar)
        plt.figure()
        shap.summary_plot(vals, X_test, plot_type="bar", show=False)
        plt.savefig(f"{config.DIRS[\"plots\"]}/shap_importance_bar.png", dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("SHAP plots saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred during SHAP value generation: {e}")

if __name__ == "__main__":
    evaluate_models()
