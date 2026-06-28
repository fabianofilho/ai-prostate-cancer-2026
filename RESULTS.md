# Results — Prostate Cancer AI / ASCO 2026

**Project:** Machine Learning for Prostate Cancer Survival Prediction Using Hospital-Based Cancer Registry Data in Brazil
**Dataset:** RHC + SIM — Espírito Santo, Brazil
**Run date:** 2026-03-29
**Last updated:** 2026-06-28 (updated with manuscript figures — main.tex PLOS Medicine)
**Pipeline:** `python run_pipeline.py --steps all`

---

## Table 1 — Study Population Characteristics

| Characteristic | Overall (N=10,552) | Alive (n=6,388) | Deceased (n=4,164) | p-value |
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

## Table 2 — Boruta-Selected Features (13 variables)

| # | Feature | Description | Domain |
|---|---------|-------------|--------|
| 1 | IDADE | Age at diagnosis (years) | Demographic |
| 2 | UN_HOSP_HUCAM | Hospital unit indicator (HUCAM) | Institutional |
| 3 | OCUPACAO_Ignorado | Unknown occupation indicator | Socioeconomic |
| 4 | ESTADIAM_1 | Clinical stage I indicator | Staging |
| 5 | ESTADIAM_4 | Clinical stage IV indicator | Staging |
| 6 | ESTADIAM_Ignorado | Unknown clinical stage indicator | Staging |
| 7 | PTNM_T2N0M0 | Pathological TNM T2N0M0 | Staging |
| 8 | PTNM_T2_outros | Other pathological T2 variants | Staging |
| 9 | PTNM_NA | Pathological TNM not applicable | Staging |
| 10 | ALCOOL_Ignorado | Unknown alcohol use indicator | Behavioral |
| 11 | CLINENTR_31.0 | Performance status 3 (bedridden <50% of day) | Clinical |
| 12 | CUSTDIAGTUMOR_NA | Diagnostic funding not applicable | Economic |
| 13 | CUSTRATAMTUMOR_NA | Treatment funding not applicable | Economic |

> Feature selection method: BorutaPy with RandomForestClassifier (max_depth=7, n_estimators=auto, random_state=42)
> Note: staging variables dominate (6 of 13); several predictors encode absence of information (unknown stage, unknown occupation, unknown alcohol use, funding not applicable), suggesting missingness itself carries prognostic signal.

---

## Table 3 — Classification Model Performance (5-year mortality, test set n=2,110)

| Model | Test F1 | AUC | Accuracy | Precision (death) | Recall (death) |
|-------|---------|-----|----------|-------------------|----------------|
| **LightGBM** | **0.515** | **0.764** | **0.751** | **0.44** | **0.61** |
| XGBoost | 0.507 | 0.764 | — | — | — |
| Random Forest | 0.502 | 0.753 | — | — | — |
| TabPFN | 0.405 | — | — | — | — |

> **Best model:** LightGBM (test F1=0.515, AUC=0.764, accuracy=0.751)
> **Classification target:** Death within 5 years of registration
> **Class distribution (test):** 1,656 alive at 5 years; 454 deaths within 5 years
> **Threshold:** 0.5 (fixed); class weighting used to handle imbalance
> **Interpretation:** Fewer than half of patients flagged as high-risk actually died within 5 years (precision=0.44), consistent with a screening-oriented classifier that favours sensitivity over precision.

---

## Table 4 — Survival Model Performance (Overall Survival, Test Set)

| Rank | Model | C-index |
|------|-------|---------|
| 1 | **Random Survival Forest** | **0.696** |
| 2 | Gradient Boosting Survival Analysis | 0.695 |
| 3 | Coxnet Survival (ElasticNet) | 0.692 |
| 3 | Extra Survival Trees | 0.692 |
| 3 | Cox Proportional Hazards | 0.692 |

> **Metric:** Harrell's C-index on held-out test set
> **Outcome:** Overall survival (time from registration to death from any cause; right-censored)
> **Key finding:** Range across models is only 0.004 — flexible tree ensembles and classical Cox model are practically indistinguishable, suggesting the prognostic signal is largely additive with the available administrative predictors.

### Time-Dependent AUC (Random Survival Forest)

| Horizon | AUC | Cases / Controls |
|---------|-----|-----------------|
| 1 year | 0.741 | 108 / 2,002 |
| 3 years | 0.751 | 319 / 1,791 |
| 5 years | 0.748 | 470 / 1,640 |
| Mean | 0.747 | — |

### Kaplan-Meier Stratified by Median Predicted Risk (RSF)

| Group | n | Deaths | Death rate |
|-------|---|--------|------------|
| High-risk (above median) | 1,055 | 597 | 56.6% |
| Low-risk (below median) | 1,055 | 235 | 22.3% |

Log-rank p = 5.76 × 10^-45

### Model Hyperparameters (Survival)

| Model | Key Parameters |
|-------|---------------|
| Random Survival Forest | n_estimators=200, min_samples_leaf=15, max_features="sqrt" |
| Extra Survival Trees | n_estimators=200, min_samples_leaf=10, max_features="sqrt" |
| **Gradient Boosting Survival** | n_estimators=200, lr=0.05, max_depth=4, subsample=0.8 |
| Coxnet Survival | l1_ratio=0.5, alpha_min_ratio=0.1, fit_baseline_model=True |
| Cox PH | alpha=0.1 (L2 regularization) |

---

## SurvSHAP(t) — Feature Importance (Random Survival Forest)

SurvSHAP(t) computed for the best survival model (RSF). Top predictors: clinical stage IV and age at diagnosis, followed by funding not applicable and unknown clinical stage. Results in:
- `results/metrics/survshap_feature_importance.csv`
- `results/plots/shap_surv_importance_bar.png`

SHAP (classification, LightGBM) converged on the same hierarchy: clinical stage IV and age as the two dominant contributors, followed by unknown/not-applicable staging and funding indicators. Convergence between two methodologically distinct explainability frameworks reinforces that these signals are intrinsic to the data.

---

## Key Findings Summary

| Finding | Value |
|---------|-------|
| Study population | 10,552 patients (RHC-ES + SIM linkage) |
| Mortality rate | 39.5% (4,164/10,552) |
| Median follow-up | 7.5 years (IQR 5.3–10.9) |
| Boruta features selected | 13 |
| Best classifier | LightGBM (F1=0.515, AUC=0.764, accuracy=75%) |
| Best survival model | Random Survival Forest (C-index=0.696) |
| Time-dependent AUC (RSF) | 0.741 (1yr), 0.751 (3yr), 0.748 (5yr) |
| KM stratification | High-risk 56.6% vs low-risk 22.3% deaths (log-rank p=5.76e-45) |
| Top predictors (SHAP + SurvSHAP) | Clinical stage IV, age at diagnosis, unknown staging, funding not applicable |

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
