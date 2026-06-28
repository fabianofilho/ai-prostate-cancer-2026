# AI for Prostate Cancer Survival Prediction
### Machine Learning Applied to Hospital-Based Cancer Registry Data in Brazil

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the complete analysis pipeline for a study on machine learning-based survival prediction in prostate cancer patients using linked hospital cancer registry (RHC) and mortality database (SIM) data from Espírito Santo, Brazil.

**Dataset:** 10,552 patients diagnosed with prostate cancer, with 4,164 deaths (39.5% overall mortality) and a median follow-up of 7.5 years (IQR 5.3–10.9).

**Authors:** Fabiano Novaes Barcellos Filho, Victor Hugo Ovani Marchetti, Wesley Rocha Grippa, Vitor Fiorin Vasconcellos, Luis Carlos Lopes-Junior

---

## Key Results

### Classification (5-year mortality prediction, test set n=2,110)

| Model | Test F1 | AUC | Accuracy |
|-------|---------|-----|----------|
| **LightGBM** | **0.515** | **0.764** | **0.751** |
| XGBoost | 0.507 | 0.764 | — |
| Random Forest | 0.502 | 0.753 | — |
| TabPFN | 0.405 | — | — |

### Survival Analysis (Harrell's C-index, overall survival)

| Rank | Model | C-index |
|------|-------|---------|
| 1 | **Random Survival Forest** | **0.696** |
| 2 | Gradient Boosting Survival | 0.695 |
| 3 | Coxnet Survival (ElasticNet) | 0.692 |
| 3 | Extra Survival Trees | 0.692 |
| 3 | Cox Proportional Hazards | 0.692 |

Time-dependent AUC (RSF): 0.741 (1 year), 0.751 (3 years), 0.748 (5 years).

**Feature selection:** 13 variables confirmed by Boruta (BorutaPy with RandomForestClassifier), including age at diagnosis, clinical staging (I, IV, unknown), pathological TNM, unknown occupation, unknown alcohol use, performance status, hospital unit (HUCAM), and funding indicators.

See [`RESULTS.md`](RESULTS.md) for complete tables with descriptive statistics and model hyperparameters.

---

## Repository Structure

```
ai-prostate-cancer-2026/
├── src/
│   ├── config.py              # Centralized configuration (paths, hyperparameters, model maps)
│   ├── preprocess.py          # Data typing, low-variance filtering, feature engineering
│   ├── select_features.py     # Boruta feature selection
│   ├── train.py               # Classification model training (LightGBM, RF, XGBoost, TabPFN)
│   ├── evaluate.py            # Evaluation, SHAP, ROC/PR curves, confusion matrices
│   ├── survival_analysis.py   # Survival model training (5 sksurv models) + SurvSHAP(t)
│   └── utils.py               # Logger and shared utilities
├── run_pipeline.py            # CLI entry point: python run_pipeline.py --steps all
├── results/
│   ├── metrics/               # CSV outputs (model comparisons, feature importances)
│   └── plots/                 # PNG figures (ROC, SHAP, survival curves, KM)
├── RESULTS.md                 # Complete results reference (Tables 1–4)
├── results_addendum_survival_extended.md
└── requirements.txt
```

> **Note:** Raw patient data (`data/`) and trained model artifacts (`results/models/*.pkl`) are not redistributed in this repository due to data governance and file-size constraints.

---

## Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (preprocessing → feature selection → training → evaluation → survival)
python run_pipeline.py --steps all

# Individual steps
python run_pipeline.py --steps preprocess
python run_pipeline.py --steps select_features
python run_pipeline.py --steps train
python run_pipeline.py --steps evaluate
python run_pipeline.py --steps survival
```

---

## Methods Summary

### Data
- **Source:** Hospital Cancer Registry of Espírito Santo (RHC-ES) linked with the Brazilian Mortality System (SIM) via probabilistic record linkage
- **Period:** Multiple diagnosis years; follow-up through 2024
- **Outcome (classification):** Binary all-cause mortality
- **Outcome (survival):** Time to death in days (right-censored), structured array via `sksurv.util.Surv`

### Preprocessing
- Semantic data typing (IBGE codes → categorical)
- Low-variance feature removal (threshold: 0.01)
- Feature engineering: age groups, diagnosis year, leakage column removal
- Missing values: median imputation (numeric), "Ignorado" (categorical)
- One-hot encoding with `drop_first=True`

### Feature Selection
- **BorutaPy** with RandomForestClassifier (max_depth=7, n_estimators="auto", random_state=42)
- 13 confirmed predictors retained

### Classification Models
- LightGBM, XGBoost, RandomForestClassifier, TabPFN
- 5-fold stratified cross-validation; threshold-optimized on F1

### Survival Models (scikit-survival)
- RandomSurvivalForest (n_estimators=200, min_samples_leaf=15, max_features="sqrt")
- ExtraSurvivalTrees (n_estimators=200, min_samples_leaf=10, max_features="sqrt")
- GradientBoostingSurvivalAnalysis (n_estimators=200, lr=0.05, max_depth=4, subsample=0.8)
- CoxnetSurvivalAnalysis (l1_ratio=0.5, ElasticNet regularization)
- CoxPHSurvivalAnalysis (L2 alpha=0.1)
- Linear models (CoxPH, Coxnet) trained with StandardScaler preprocessing

### Explainability
- **SHAP** (classification): TreeExplainer, beeswarm + bar plots
- **SurvSHAP(t)** (survival): MI2DataLab `survshap` library, individual PredictSurvSHAP loop, sampling method, B=25 permutations

---

## Dependencies

```
scikit-survival>=0.22.0    # Survival models (RSF, GBS, Coxnet, EST, CoxPH)
survshap                   # SurvSHAP(t) explanations
lightgbm>=4.0.0
xgboost>=2.0.0
tabpfn>=0.1.9
boruta>=0.3.0
shap>=0.44.0
lifelines>=0.27.0          # Kaplan-Meier plots
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

See `requirements.txt` for the full pinned dependency list.

---

## Citation

*Manuscript under review. Citation information will be updated upon publication.*

If you use this code or results, please cite:

> Barcellos Filho FN, Marchetti VHO, Grippa WR, Vasconcellos VF, Lopes-Junior LC. Machine learning for prostate cancer survival prediction using hospital-based cancer registry data in Brazil: a retrospective cohort study. *[Journal name upon acceptance]*, 2026.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

> Patient-level data are not included and are subject to ethics approval and Brazilian data protection legislation (LGPD). Access requests should be directed to the corresponding author.
