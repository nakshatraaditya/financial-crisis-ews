# Simulation Tool for Economic Crisis Prediction  
*A Financial Crisis Early Warning System (EWS) for the United States and the United Kingdom (1870–2020)*

This repository contains a **policy-oriented simulation tool** for estimating the probability of systemic banking crises using long-run macro-financial indicators and market-based behavioural proxies. The project is developed as part of an **MSc Data Science dissertation** and is designed to reflect how Early Warning Systems (EWS) are used in real macroprudential surveillance—to flag periods of elevated systemic risk rather than to predict exact crisis years with certainty.

---

## Project Overview

Financial crises are rare, endogenous, and highly disruptive. This project formulates crisis prediction as a **forward-looking supervised classification problem**, where each year is assigned a probability of a crisis occurring within a fixed forecast horizon.

Rather than maximising raw accuracy, the system emphasises **operational usefulness**:
- timely detection of rising systemic risk,
- constrained alerting under limited supervisory capacity,
- interpretability and robustness suitable for policy environments.

Alerts are generated using an **alert budget** (e.g. flag only the top 20% highest-risk years), reflecting the reality that policymakers cannot respond to every warning signal.

---

## Key Design Principles

The workflow prioritises:

- **Time-consistent validation**  
  No future information is used in preprocessing, scaling, or threshold selection.

- **Leakage-safe feature engineering**  
  All transformations respect the information set available at each point in time.

- **Rare-event evaluation**  
  Performance is assessed using event-level recall, PR-AUC, and calibration metrics rather than accuracy.

- **Interpretability and transparency**  
  Model outputs are explained using SHAP-style feature attribution and structured ablation analysis.

---

## Data

This study uses the **JST Macrohistory Database (Release 6)** and its associated financial crisis chronology.

- **Countries:** United States, United Kingdom  
- **Frequency:** Annual  
- **Period:** 1870–2020 (subject to variable availability)  
- **Target:** Forward-looking systemic banking crisis indicator  

The crisis labels are based on the JST systemic banking crisis chronology, which is widely used in the financial stability literature.

> Data access and documentation:  
> https://www.macrohistory.net/database/

---

## Method Summary

### 1) Target Construction (Forecast Horizon)
The target variable is forward-looking: a year is labelled positive if a crisis occurs within the next *H* years (typically *H = 2*).  
This ensures the model generates **early warnings**, not contemporaneous crisis detection.

### 2) Cleaning and Transformations (Leakage-Safe)
Historical macro-financial data contains substantial missingness, especially in early periods. The pipeline uses:
- forward-fill within country (short, capped window),
- **training-only median imputation** for remaining gaps,
- explicit **missingness indicators** to preserve information in data availability patterns.

Backward filling was explicitly avoided due to look-ahead bias.

### 3) Standardisation
Continuous features are z-score standardised using **training-set statistics only**.  
The same preprocessing logic is applied across models to ensure comparability.

### 4) Modelling
Models explored include:
- **Logistic Regression** (econometric benchmark and final reference model),
- Random Forest,
- Gradient Boosting,
- Support Vector Machine,
- Neural Network (MLP).

Final emphasis is placed on **out-of-sample stability, interpretability, and policy defensibility**, rather than marginal performance gains.

### 5) Class Imbalance Handling
Systemic crises are rare. The framework therefore focuses on:
- class-aware training (where applicable),
- probability thresholding via alert budgets,
- **event-level evaluation** rather than point-wise accuracy.

### 6) Evaluation
Reported metrics include:
- **Event-level recall** (whether each crisis is preceded by at least one alert),
- **PR-AUC** (ranking quality under imbalance),
- **Brier score** (probability calibration),
- realised alert rates under fixed budgets.

### 7) Interpretability and Robustness
- **SHAP explanations** are used to identify global and local drivers of predicted risk.
- An **ablation study** evaluates the contribution of:
  - macro-financial indicators,
  - behavioural / market-based proxies,
  - explicit missingness indicators.

---

## Repository Structure
├── app.py # Streamlit dashboard (interactive EWS)
├── JSTdatasetR6.xlsx 
├── requirements.txt 
├── 240377687_Nakshatra_Aditya.ipynb
└── README.md

---

## Running the Dashboard Locally

The project includes an interactive **Streamlit dashboard** for exploring crisis risk, alerts, robustness checks, ablation results, and SHAP explanations.

### Prerequisites
- **Python 3.9+**
- Git
- JST Macrohistory Database (Release 6)

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/-An-Early-Warning-System-for-Systemic-Banking-Crises-.git
cd -An-Early-Warning-System-for-Systemic-Banking-Crises-
