# -*- coding: utf-8 -*-
"""
Survival analysis script using scikit-survival (sksurv).

This script implements the next phase of the project, going beyond binary
classification to model time-to-event data. It trains and evaluates multiple
survival models (RSF, ExtraSurvivalTrees, GradientBoostingSurvival, Coxnet,
CoxPH) and generates SurvSHAP(t) explanations for the best-performing model.

NOTE: This script requires the raw data with the 'TEMPO_DIAS' (follow-up time)
and 'STATUS' (event indicator) columns, which are removed in the classification
pipeline due to leakage. The survival pipeline uses them as the outcome.

Requirements:
    pip install scikit-survival survshap
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

from src import config
from src.utils import get_logger

logger = get_logger(__name__)


def prepare_survival_data(df):
    """
    Prepares the dataset for survival analysis.

    Reuses the same preprocessing as the classification pipeline but retains
    TEMPO_DIAS and STATUS for the structured survival outcome. Uses the Boruta-
    selected features so that the survival models are comparable to the
    classification results and train in reasonable time.
    """
    logger.info("Preparing data for survival analysis...")

    # --- 1. Build the structured outcome BEFORE preprocessing drops the columns ---
    df = df.copy()
    df["TEMPO_DIAS"] = pd.to_numeric(df["TEMPO_DIAS"], errors="coerce")
    df = df.dropna(subset=["TEMPO_DIAS", "STATUS"])
    df = df[df["TEMPO_DIAS"] > 0]

    event = df["STATUS"].map({"ALIVE": False, "PCA_DEATH": True, "OTHER_DEATH": True}).astype(bool)
    time = df["TEMPO_DIAS"].astype(float)
    y_surv = Surv.from_arrays(event=event.values, time=time.values)

    # --- 2. Run the same preprocessing as the classification pipeline ---
    from src.preprocess import apply_data_types, remove_low_variance_features, feature_engineering

    df = apply_data_types(df)
    df = remove_low_variance_features(df)
    # feature_engineering drops leakage cols (STATUS, TEMPO_DIAS, etc.)
    df = feature_engineering(df)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df[cat_cols] = df[cat_cols].fillna("Ignorado")
    df = pd.get_dummies(df, drop_first=True)
    df = df.fillna(df.median())

    # --- 3. Select Boruta features (same 16 used in classification) ---
    try:
        selected = pd.read_csv(f"{config.DIRS['metrics']}/features_selected.csv", header=0).iloc[:, 0].tolist()
        available = [f for f in selected if f in df.columns]
        if len(available) >= 5:
            df = df[available]
            logger.info(f"Using {len(available)} Boruta-selected features for survival analysis.")
        else:
            logger.warning(f"Only {len(available)} Boruta features found. Using all {df.shape[1]} features.")
    except FileNotFoundError:
        logger.warning("Boruta features file not found. Using all features.")

    # Align X rows with y_surv (they share the same index from the initial filter)
    X = df.loc[event.index]

    logger.info(f"Survival data prepared. Shape: {X.shape}, Events: {event.sum()}/{len(event)}")
    return X, y_surv


def train_and_evaluate_survival_models(X, y_surv):
    """
    Trains, evaluates, and compares survival models defined in config.SURVIVAL_MODELS_MAP.

    Metrics:
    - Harrell's C-index (concordance index) — main ranking metric
    - Integrated Brier Score (IBS) — calibration metric, where applicable

    Tree ensemble models (RSF, ExtraSurvivalTrees, GBM) use raw features.
    Linear models (CoxPH, CoxnetSurvival) use StandardScaler-normalized features.
    """
    logger.info("Splitting data for survival analysis...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_surv, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # --- Fit scaler once (used only for models in SURVIVAL_MODELS_NEED_SCALING) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{config.DIRS['models']}/survival_scaler.pkl")

    results = {}

    for name, model in config.SURVIVAL_MODELS_MAP.items():
        logger.info(f"Training {name}...")
        Xtr = X_train_scaled if name in config.SURVIVAL_MODELS_NEED_SCALING else X_train
        Xte = X_test_scaled if name in config.SURVIVAL_MODELS_NEED_SCALING else X_test

        # Convert scaled arrays back to DataFrame to preserve feature names
        if isinstance(Xtr, np.ndarray):
            Xtr = pd.DataFrame(Xtr, columns=X_train.columns, index=X_train.index)
            Xte = pd.DataFrame(Xte, columns=X_test.columns, index=X_test.index)

        model.fit(Xtr, y_train)
        cindex = model.score(Xte, y_test)

        # --- Integrated Brier Score (where predict_survival_function is available) ---
        ibs = None
        try:
            # Use 10th–90th percentile of test times to stay within train range
            test_times = np.array([t for _, t in y_test])
            times = np.percentile(test_times, np.linspace(10, 90, 20))
            surv_fns = model.predict_survival_function(Xte)
            preds = np.row_stack([fn(times) for fn in surv_fns])
            _, ibs = integrated_brier_score(y_train, y_test, preds, times)
        except Exception:
            pass

        results[name] = {
            "C-index": round(cindex, 4),
            "IBS": round(ibs, 4) if ibs is not None else None,
        }
        joblib.dump(model, f"{config.DIRS['models']}/{name}.pkl")
        ibs_str = f", IBS={ibs:.4f}" if ibs is not None else ""
        logger.info(f"  {name}: C-index={cindex:.4f}{ibs_str}")

    # --- Comparison table ---
    df_results = (
        pd.DataFrame(results)
        .T.reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("C-index", ascending=False)
        .reset_index(drop=True)
    )
    df_results.to_csv(f"{config.DIRS['metrics']}/survival_model_comparison.csv", index=False)
    logger.info("Survival model comparison saved.")
    print("\n" + df_results.to_string(index=False))

    # --- Comparison bar chart (C-index) ---
    _plot_survival_comparison(df_results)

    # --- SurvSHAP(t): try best model, fall back to best tree ensemble ---
    # survshap works best with tree-based models (RSF, ExtraSurvivalTrees);
    # GradientBoostingSurvival may have incompatible prediction interface.
    SHAP_PREFERRED_ORDER = ["RandomSurvivalForest", "ExtraSurvivalTrees",
                            "GradientBoostingSurvival", "CoxnetSurvival", "CoxPH"]
    shap_candidates = [m for m in SHAP_PREFERRED_ORDER if m in df_results["Model"].values]
    # Prefer best C-index, but fall back to tree models if survshap fails later
    shap_candidates = sorted(
        shap_candidates,
        key=lambda m: (-df_results.set_index("Model").loc[m, "C-index"],
                       SHAP_PREFERRED_ORDER.index(m))
    )
    best_name = shap_candidates[0]
    best_model = config.SURVIVAL_MODELS_MAP[best_name]
    Xtr_best = X_train_scaled if best_name in config.SURVIVAL_MODELS_NEED_SCALING else X_train
    Xte_best = X_test_scaled if best_name in config.SURVIVAL_MODELS_NEED_SCALING else X_test
    if isinstance(Xtr_best, np.ndarray):
        Xtr_best = pd.DataFrame(Xtr_best, columns=X_train.columns, index=X_train.index)
        Xte_best = pd.DataFrame(Xte_best, columns=X_test.columns, index=X_test.index)
    logger.info(f"Best model: {best_name} (C-index={df_results.set_index('Model').loc[best_name, 'C-index']:.4f})")
    success = generate_shap_survival(best_model, Xtr_best, Xte_best, y_train, y_test)

    # Fallback: if best model failed SurvSHAP, retry with first available RSF/EST
    if not success:
        fallback_order = ["RandomSurvivalForest", "ExtraSurvivalTrees"]
        for fb_name in fallback_order:
            if fb_name in config.SURVIVAL_MODELS_MAP and fb_name != best_name:
                logger.info(f"SurvSHAP fallback: trying {fb_name}...")
                fb_model = config.SURVIVAL_MODELS_MAP[fb_name]
                if generate_shap_survival(fb_model, X_train, X_test, y_train, y_test):
                    break

    return df_results, X_train, X_test


def _plot_survival_comparison(df_results):
    """Generates a horizontal bar chart comparing C-index across survival models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [config.SURVIVAL_PLOT_COLORS.get(m, "gray") for m in df_results["Model"]]
    bars = ax.barh(df_results["Model"], df_results["C-index"], color=colors)
    ax.set_xlabel("C-index (Harrell's concordance)")
    ax.set_title("Survival Model Comparison — C-index (Test Set)")
    ax.set_xlim(0.5, max(df_results["C-index"]) + 0.05)
    ax.invert_yaxis()
    for bar, val in zip(bars, df_results["C-index"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{config.DIRS['plots']}/survival_model_comparison.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Survival comparison plot saved.")


def generate_shap_survival(model, X_train, X_test, y_train, y_test):
    """
    Generates SurvSHAP(t) explanations for the best survival model.

    Uses the survshap library (https://github.com/MI2DataLab/survshap) which
    provides time-dependent SHAP values specifically designed for survival models.
    Individual PredictSurvSHAP loop avoids aggregation shape mismatch in ModelSurvSHAP.
    """
    logger.info("Generating SurvSHAP(t) values for the best survival model...")
    try:
        from survshap import SurvivalModelExplainer, PredictSurvSHAP

        # Sample background for speed
        bg_idx = np.random.RandomState(config.RANDOM_STATE).choice(
            len(X_train), size=min(500, len(X_train)), replace=False
        )
        explainer = SurvivalModelExplainer(
            model=model,
            data=X_train.iloc[bg_idx],
            y=y_train[bg_idx],
        )

        # Timestamps: use unique_times_ for tree ensembles; percentiles otherwise
        if hasattr(model, "unique_times_"):
            timestamps = model.unique_times_[:500]
        else:
            train_times = np.array([t for _, t in y_train])
            timestamps = np.percentile(train_times, np.linspace(5, 95, 100))

        n_sample = min(25, len(X_test))
        logger.info(f"Computing SurvSHAP(t) for {n_sample} test observations (sampling method)...")

        all_importances = []
        for i in range(n_sample):
            if i % 5 == 0:
                logger.info(f"  SurvSHAP observation {i + 1}/{n_sample}...")
            try:
                pred_shap = PredictSurvSHAP(
                    calculation_method="sampling",
                    random_state=config.RANDOM_STATE,
                    B=25,
                )
                pred_shap.fit(
                    explainer=explainer,
                    new_observation=X_test.iloc[[i]],
                    timestamps=timestamps,
                )
                result = pred_shap.result
                if result is not None and "aggregated_change" in result.columns:
                    imp = result.set_index("variable_name")["aggregated_change"]
                    all_importances.append(imp)
            except Exception:
                continue

        if len(all_importances) > 0:
            importance_df = pd.DataFrame(all_importances)
            mean_importance = importance_df.apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)

            # Save CSV
            mean_importance.to_csv(f"{config.DIRS['metrics']}/survshap_feature_importance.csv")
            logger.info("SurvSHAP feature importances saved.")
            print("\nSurvSHAP(t) Feature Importance (mean |SHAP|):")
            print(mean_importance.to_string())

            # Bar plot
            plt.figure(figsize=(10, 6))
            mean_importance.head(16).plot(kind="barh", color="steelblue")
            plt.xlabel("Mean |SurvSHAP(t)| value")
            plt.title("SurvSHAP(t) Feature Importance — Best Survival Model")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                f"{config.DIRS['plots']}/shap_surv_importance_bar.png",
                dpi=config.FIGURE_DPI,
                bbox_inches="tight",
            )
            plt.close()
            logger.info("SurvSHAP bar plot saved.")
        else:
            logger.warning("No SurvSHAP results computed successfully.")

        logger.info("SurvSHAP(t) analysis completed.")
        return len(all_importances) > 0

    except Exception as e:
        logger.error(f"Error during SurvSHAP(t) generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main survival analysis pipeline."""
    try:
        df = pd.read_csv(config.RAW_DATA_FILE, sep=";", na_values=["9", "99", "999"])
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {config.RAW_DATA_FILE}")
        raise

    X, y_surv = prepare_survival_data(df)
    train_and_evaluate_survival_models(X, y_surv)
    logger.info("Survival analysis pipeline completed.")


if __name__ == "__main__":
    main()
