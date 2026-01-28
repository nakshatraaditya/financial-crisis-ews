# Simulation Tool for Economic Crisis Prediction  
*A Financial Crisis Early Warning System (EWS) for the United States and the United Kingdom (1870–2020).*

This repository contains a simulation-based Early Warning System designed to estimate the probability of systemic banking crises using long-run macro-financial indicators and market-based behavioural proxies. The project is aligned with policy-style monitoring: the goal is to generate timely warning signals ahead of crisis events, not to predict exact crisis years with certainty.

---

## Project Overview

Financial crises are rare but highly disruptive. This project frames crisis prediction as a forward-looking supervised classification task. Using annual historical data, the system produces a crisis-risk score each year and converts it into alerts under a constrained *alert budget* (e.g., flag only the top 20% highest-risk years), reflecting the operational reality that policymakers can only investigate a limited number of warnings.

The workflow prioritises:
- **Realistic, time-consistent validation** (rolling-origin evaluation)
- **Leakage-safe preprocessing** (no use of future information)
- **Rare-event evaluation metrics** (event-based crisis capture, PR-AUC, calibration)
- **Interpretability** (SHAP-style feature attribution)

---

## Data

This study uses the **JST Macrohistory Database (Release 6)** and its associated crisis chronology.

- Countries: **United States, United Kingdom**
- Frequency: **Annual**
- Period: **1870–2020** (subject to variable availability)
- Target: **Forward-looking crisis indicator** (crisis occurs within the next *H* years)

> Data access and documentation: JST Macrohistory Database (Release 6) via macrohistory.net.

---

## Method Summary

### 1) Target Construction (Forecast Horizon)
The label is forward-looking: a year is positive if a crisis occurs within the next *H* years.
This ensures warnings precede crisis onset rather than simply identifying crisis years.

### 2) Cleaning and Transformations (Leakage-safe)
Long-run historical data contains missingness, especially in early periods. The final pipeline uses:
- **Forward-fill within country with a short limit**
- **Training-only median imputation** for remaining missing values
- **Missingness flags** to preserve information in data availability patterns  
Backward filling was tested early on but excluded due to look-ahead bias (data leakage).

Nominal variables are converted into real terms where required, and features are expressed as:
- growth rates
- ratios
- spreads
- historically grounded deviations

### 3) Standardisation
Continuous features are z-score standardised using **training-only** parameters to avoid look-ahead bias.
A consistent preprocessing pipeline is applied across model classes for comparability.

### 4) Modelling
Models explored include:
- **Econometric benchmark:** Logistic Regression (baseline and final selection)
- **Non-linear ML models:** Random Forest, Gradient Boosting, Neural Network, SVM (tested as complements)

Final selection prioritised out-of-sample stability under rolling-origin validation and policy-style alerting.

### 5) Class Imbalance Handling
Crises are rare, so evaluation focuses on policy-relevant performance rather than raw accuracy.
The system uses:
- class-aware training strategies (where applicable)
- thresholding via **alert budget**
- event-level evaluation

### 6) Evaluation
Performance is reported using complementary metrics:
- **Event-level recall**: whether each crisis has at least one pre-crisis alert
- **PR-AUC**: ranking quality under class imbalance
- **Brier score**: probability calibration quality
- **Alert-rate behaviour** under fixed alert budgets

### 7) Interpretability
SHAP-style explanations are used to quantify feature contributions:
- global importance (mean absolute SHAP)
- local explanations for pre-crisis periods

An **ablation study** tests whether performance depends on specific feature families (macro vs behavioural) and whether missingness indicators materially contribute.

---

## Repository Structure (suggested)

