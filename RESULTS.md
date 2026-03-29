# Results — Prostate Cancer AI / ASCO 2026

**Project:** Machine Learning for Prostate Cancer Survival Prediction Using Hospital-Based Cancer Registry Data in Brazil
**Dataset:** RHC + SIM — Espírito Santo, Brazil
**Run date:** 2026-03-29
**Last updated:** 2026-03-29 (Table 3 completed with full precision/sensitivity for all models)
**Pipeline:** `python run_pipeline.py --steps all`

---

## Table 1 — Study Population Characteristics

| Characteristic | Overall (N=10,550) | Alive (n=6,388) | Deceased (n=4,162) | p-value |
|---|---|---|---|---|
| **Age at diagnosis, mean (SD)** | 69.2 (9.1) | 67.3 (8.7) | 72.1 (8.9) | <0.001 |
| **Age group, n (%)** | | | | <0.001 |
| < 60 years | 1,816 (17.2%) | 1,387 (21.7%) | 429 (10.3%) | |
| 60–69 years | 3,872 (36.7%) | 2,625 (41.1%) | 1,247 (30.0%) | |
| 70–79 years | 3,846 (36.5%) | 2,019 (31.6%) | 1,827 (43.9%) | |
| ≥80 years | 1,016 (9.6%) | 357 (5.6%) | 659 (15.8%) | |
| **Race/ethnicity, n (%)** *(non-missing, n=9,698)* | | | | — |
| White | 3,007 (31.0%) | 1,799 (28.2%) | 1,208 (29.0%) | |
| Brown (Pardo) | 5,948 (61.3%) | 3,620 (56.7%) | 2,328 (55.9%) | |
| Black | 675 (7.0%) | 446 (7.0%) | 229 (5.5%) | |
| Asian / Indigenous | 68 (0.7%) | 39 (0.6%) | 29 (0.7%) | |
| **Clinical stage, n (%)** *(non-missing, n=5,492)* | | | | — |
| Stage I | 735 (13.4%) | 555 (10.1%) | 180 (4.3%) | |
| Stage II | 2,542 (46.3%) | 1,586 (24.8%) | 956 (23.0%) | |
| Stage III | 1,014 (18.5%) | 641 (10.0%) | 373 (9.0%) | |
| Stage IV | 1,093 (19.9%) | 271 (4.2%) | 822 (19.8%) | |
| Unknown | 5,058 (47.9%) | — | — | |
| **Smoking status, n (%)** *(non-missing, n=6,309)* | | | | — |
| Never | 1,987 (31.5%) | — | — | |
| Current | 1,149 (18.2%) | — | — | |
| Former | 969 (15.4%) | — | — | |
| Unknown | 2,204 (34.9%) | — | — | |
| **Follow-up time (days), median (IQR)** | 2,726 (1,922–3,986) | 3,284 (2,433–4,511) | 1,632 (744–2,794) | <0.001 |

> **Notes:**
> - p-values: t-test for continuous variables; chi-square for categorical
> - Race/ethnicity missingness: 852 patients (8.1%)
> - Clinical stage missingness: 5,058 patients (47.9%) — RHC coding gap common in public hospitals
> - Smoking missingness: 4,241 patients (40.2%)

---

## Table 2 — Boruta-Selected Features (16 variables)

| # | Feature | Description | Domain | Missingness |
|---|---------|-------------|--------|-------------|
| 1 | IDADE | Age at diagnosis (years) | Demographics | 0.0% |
| 2 | ESTADIAM_1 | Clinical stage I indicator | Clinical staging | 47.9% |
| 3 | ESTADIAM_4 | Clinical stage IV indicator | Clinical staging | 47.9% |
| 4 | ESTADIAM_Ignorado | Unknown clinical stage indicator | Clinical staging | 47.9% |
| 5 | PTNM_T2N0M0 | Pathological TNM category T2N0M0 | Pathological staging | 26.3% |
| 6 | PTNM_T2NxMx | Pathological TNM category T2NxMx | Pathological staging | 26.3% |
| 7 | PTNM_T2_outros | Other pathological T2 variants | Pathological staging | 26.3% |
| 8 | PTNM_NA | Pathological TNM not applicable | Pathological staging | 26.3% |
| 9 | RAZAONT_2.0 | Race/ethnicity category indicator (Black) | Sociodemographic | 8.1% |
| 10 | RAZAONT_6.0 | Race/ethnicity category indicator (other) | Sociodemographic | 8.1% |
| 11 | TABAGISM_3.0 | Former smoker status indicator | Behavioral | 40.0% |
| 12 | CLINENTR_31.0 | Poor clinical condition at presentation | Clinical condition | 0.3% |
| 13 | CUSTDIAGTUMOR_8.0 | Diagnostic cost-category indicator | Economic | 1.8% |
| 14 | CUSTRATAMTUMOR_8.0 | Treatment cost-category indicator | Economic | 3.2% |
| 15 | UN_HOSP_HUCAM | Hospital unit indicator (HUCAM) | Institutional | 0.0% |
| 16 | OCUPACAO_Ignorado | Unknown occupation indicator | Socioeconomic | 30.3% |

> Feature selection method: BorutaPy with RandomForestClassifier (max_depth=7, n_estimators=auto, random_state=42)

---

## Table 3 — Classification Model Performance (Test Set, n=2,110)

| Model | Test F1 | CV F1 (mean) | Accuracy | Precision (death) | Sensitivity (death) |
|-------|---------|--------------|----------|-------------------|---------------------|
| **LightGBM** | **0.6697** | **0.6275** | **0.722** | **0.630** | **0.715** |
| Random Forest | 0.6652 | 0.5670 | 0.719 | 0.628 | 0.707 |
| XGBoost | 0.6610 | 0.6236 | 0.697 | 0.592 | 0.749 |
| TabPFN | 0.6055 | 0.5752 | — | — | — |

> **Best model:** LightGBM (highest F1 and accuracy)
> **Classification target:** Binary mortality (PCA death + other death vs. alive)
> **Class distribution:** 1,278 alive (60.6%), 832 deceased (39.4%) in test set
> **Note:** TabPFN v2.6 requires gated HuggingFace model; precision/sensitivity from previous authenticated run.
>
> Full classification reports (positive class = Óbito/death):

**LightGBM:**
```
              precision  sensitivity  f1-score   support
        Vivo       0.80       0.73      0.76      1278
       Óbito       0.63       0.72      0.67       832
    accuracy                            0.72      2110
   macro avg       0.71       0.72      0.71      2110
weighted avg       0.73       0.72      0.72      2110
```

**Random Forest:**
```
              precision  sensitivity  f1-score   support
        Vivo       0.79       0.73      0.76      1278
       Óbito       0.63       0.71      0.67       832
    accuracy                            0.72      2110
   macro avg       0.71       0.72      0.71      2110
weighted avg       0.73       0.72      0.71      2110
```

**XGBoost:**
```
              precision  sensitivity  f1-score   support
        Vivo       0.80       0.66      0.73      1278
       Óbito       0.59       0.75      0.66       832
    accuracy                            0.70      2110
   macro avg       0.70       0.71      0.69      2110
weighted avg       0.72       0.70      0.70      2110
```

---

## Table 4 — Survival Model Performance (Test Set)

| Ranking | Model | Family | C-index | vs. CoxPH baseline |
|---------|-------|--------|---------|---------------------|
| 1 | **Gradient Boosting Survival Analysis** | Sequential boosting | **0.7093** | +0.0047 |
| 2 | Random Survival Forest | Parallel ensemble (optimized) | 0.7070 | +0.0024 |
| 3 | Coxnet Survival (ElasticNet) | Penalized Cox | 0.7059 | +0.0013 |
| 4 | Cox Proportional Hazards | Linear (Cox classic) | 0.7046 | baseline |
| 5 | Extra Survival Trees | Parallel ensemble (random) | 0.7045 | −0.0001 |

> **Metric:** Harrell's C-index (concordance index) on held-out test set
> **Test set:** n=2,110 patients, 832 events (39.4%)
> **Consistent finding:** Boosting family (GBS) outperforms parallel ensembles (RSF, EST), mirroring the LightGBM > Random Forest advantage observed in classification

### Model Hyperparameters (Survival)

| Model | Key Parameters |
|-------|---------------|
| Random Survival Forest | n_estimators=200, min_samples_leaf=15, max_features="sqrt" |
| Extra Survival Trees | n_estimators=200, min_samples_leaf=10, max_features="sqrt" |
| **Gradient Boosting Survival** | n_estimators=200, lr=0.05, max_depth=4, subsample=0.8 |
| Coxnet Survival | l1_ratio=0.5, alpha_min_ratio=0.1, fit_baseline_model=True |
| Cox PH | alpha=0.1 (L2 regularization) |

---

## SurvSHAP(t) — Feature Importance (Gradient Boosting Survival)

SurvSHAP(t) computed for the best survival model (GBS) using 25 test observations,
B=25 permutations, sampling method. Results in:
- `results/metrics/survshap_feature_importance.csv`
- `results/plots/shap_surv_importance_bar.png`

---

## Key Findings Summary

| Finding | Value |
|---------|-------|
| Study population | 10,550 patients (RHC-ES + SIM linkage) |
| Mortality rate | 39.5% (4,162/10,550) |
| Median follow-up | 7.5 years (IQR 5.3–10.9) |
| Boruta features selected | 16 |
| Best classifier | LightGBM (F1=0.6697, Accuracy=72%) |
| Best survival model | Gradient Boosting Survival (C-index=0.7093) |
| Top survival predictors | Clinical staging, pathological TNM, race, treatment cost indicators |

---

## Reproducibility

```bash
# Full pipeline
python run_pipeline.py --steps all

# Only survival analysis (fastest)
python run_pipeline.py --steps survival

# Environment
pip install scikit-survival survshap lightgbm xgboost tabpfn boruta
```

See `requirements.txt` for full dependency list.
