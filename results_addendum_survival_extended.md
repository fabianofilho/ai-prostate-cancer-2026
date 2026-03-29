# Results Addendum — Extended Survival Analysis
**Projeto:** Prostate Cancer AI — ASCO 2026 / PLOS Medicine
**Data:** 2026-03-29
**Autores:** Fabiano Novaes Barcellos Filho, Wesley Rocha Grippa et al.
**Repositório:** https://github.com/fabianofilho/prostate-cancer-ai-asco2026

---

## Contexto

Este addendum complementa o paper draft principal com os resultados da análise de
sobrevivência expandida. Além dos dois modelos originais (RSF e Cox PH), adicionamos
três modelos de famílias diferentes para comparação sistemática.

Dataset: 10.550 pacientes, 4.162 eventos (39,4% de mortalidade).
Divisão treino/teste: 80/20, random_state=42.
Features: 16 variáveis selecionadas pelo Boruta.

---

## Resultados — Modelos de Sobrevivência (Test Set)

| Ranking | Modelo | Família | C-index | Melhoria vs CoxPH baseline |
|---------|--------|---------|---------|---------------------------|
| 1 | **Gradient Boosting Survival** | Boosting sequencial | **0.7093** | +0.0047 |
| 2 | Random Survival Forest | Ensemble (otimizado) | 0.7070 | +0.0024 |
| 3 | Coxnet Survival (ElasticNet) | Cox penalizado | 0.7059 | +0.0013 |
| 4 | Cox Proportional Hazards | Linear (Cox clássico) | 0.7046 | baseline |
| 5 | Extra Survival Trees | Ensemble (aleatório) | 0.7045 | -0.0001 |

> **Melhor modelo:** Gradient Boosting Survival Analysis — C-index = **0.7093**
> **Melhoria sobre RSF** (melhor anterior): +0.0023 pontos

---

## Configuração dos Modelos

### Random Survival Forest (baseline — paper draft v2)
- `n_estimators=200`, `min_samples_leaf=15`, `min_samples_split=10`
- `max_features="sqrt"`, `random_state=42`

### Extra Survival Trees (novo)
- `n_estimators=200`, `min_samples_leaf=10`, `min_samples_split=6`
- `max_features="sqrt"`, `random_state=42`
- Splits em pontos aleatórios (não otimizados) → ligeiramente inferior ao RSF neste dataset

### Gradient Boosting Survival Analysis (novo — MELHOR MODELO)
- `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`
- `min_samples_split=10`, `subsample=0.8`, `random_state=42`
- Otimiza diretamente a log-verossimilhança parcial de Cox em regime de boosting sequencial
- SurvSHAP(t) aplicado a este modelo como melhor por C-index

### Coxnet Survival — ElasticNet Cox (novo)
- `l1_ratio=0.5` (balanceia seleção Lasso + estabilidade Ridge)
- `alpha_min_ratio=0.1`, `max_iter=1000`, `fit_baseline_model=True`
- StandardScaler aplicado antes do fit
- Supera CoxPH padrão (+0.0013 C-index) com regularização ElasticNet

### Cox Proportional Hazards (baseline — paper draft v2)
- `alpha=0.1` (penalização L2 leve)
- StandardScaler aplicado antes do fit

---

## Achado Principal

O **Gradient Boosting Survival Analysis** obteve o melhor desempenho (C-index = 0.7093),
consistente com a vantagem de modelos de boosting observada na etapa de classificação
(LightGBM superou Random Forest também na classificação binária do desfecho).

Este padrão sugere que a heterogeneidade não-linear nos dados RHC é melhor capturada
por aprendizado sequencial (boosting) do que por ensembles paralelos (Random Forest).

---

## SurvSHAP(t) — Modelo Gradient Boosting Survival

SurvSHAP(t) aplicado ao melhor modelo. Os valores de importância de features
representam a contribuição média ao longo do tempo (mean |SurvSHAP|) para 25
observações de teste com B=25 permutações por observação.

> Resultado em `results/metrics/survshap_feature_importance.csv`
> Plot em `results/plots/shap_surv_importance_bar.png`

---

## Atualizações Necessárias no Paper Draft

### Abstract
- Substituir: "Random Survival Forest achieved a concordance index of 0.7070 and Cox PH 0.7046"
- Por: "Five survival models were compared: Gradient Boosting Survival Analysis achieved the highest concordance index (C-index = 0.7093), followed by Random Survival Forest (C-index = 0.7070), Coxnet Survival (0.7059), Cox PH (0.7046), and Extra Survival Trees (0.7045)"

### Methods — Section 2.5 (Survival Analysis)
Substituir parágrafo de 2 modelos por:

> "Five survival models were trained and evaluated using the same 16 Boruta-selected
> features: (1) Random Survival Forest (RSF; 200 trees, min_samples_leaf=15,
> max_features='sqrt'); (2) Extra Survival Trees (EST; identical to RSF but with
> random split thresholds); (3) Gradient Boosting Survival Analysis (GBS;
> n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8); (4) Cox
> Regression with ElasticNet regularization (Coxnet; l1_ratio=0.5); (5) Standard
> Cox Proportional Hazards (CoxPH; L2 alpha=0.1). Tree ensemble models were
> trained on unscaled features; linear models received StandardScaler preprocessing.
> Performance was assessed via Harrell's C-index on the held-out test set."

### Results — Table 4
Substituir tabela de 2 linhas por tabela de 5 linhas conforme acima.

### Discussion
Mencionar que o padrão boosting > ensemble paralelo se repete tanto na classificação
(LightGBM > RF) quanto na análise de sobrevivência (GBS > RSF).

---

*Addendum gerado em 2026-03-29. Pipeline rodado com python run_pipeline.py --steps survival.*
