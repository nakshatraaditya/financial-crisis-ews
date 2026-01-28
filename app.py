
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# Optional SHAP
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier


# =============================================================================
# STREAMLIT CONFIG
# =============================================================================
st.set_page_config(
    page_title="Financial Crisis EWS — Dissertation Dashboard",
    page_icon="📉",
    layout="wide",
)

APP_DIR = Path(__file__).parent
JST_XLSX = APP_DIR / "JSTdatasetR6.xlsx"

st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2.5rem; }
.small-note { font-size: 0.88rem; opacity: 0.85; }
.kpi-card { padding: 0.75rem 0.85rem; border: 1px solid rgba(120,120,120,0.25); border-radius: 14px; }
hr { margin: 0.6rem 0 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CONSTANTS (USA + UK ONLY)
# =============================================================================
COUNTRIES_FIXED = ["USA", "UK"]

TRAIN_END = 1970
VAL_START, VAL_END = 1970, 1990  # VAL is [1970, 1990)
TEST_START = 1990

HORIZON_GRID = [1, 2, 3]
BUDGET_GRID = [0.10, 0.20, 0.30]


# =============================================================================
# UTILITIES
# =============================================================================
COUNTRY_ORDER = ["USA", "UK"]
COUNTRY_RGB = {
    "USA": "rgb(255, 99, 71)",      # tomato
    "UK": "rgb(100, 149, 237)",     # cornflowerblue
}

def stable_rgb(name: str) -> str:
    h = abs(hash(name))
    r = 60 + (h % 140)
    g = 60 + ((h // 140) % 140)
    b = 60 + ((h // (140 * 140)) % 140)
    return f"rgb({r},{g},{b})"

def build_color_scale(categories):
    cats = [str(c) for c in categories]
    domain = []
    for c in COUNTRY_ORDER:
        if c in cats and c not in domain:
            domain.append(c)
    for c in sorted(set(cats)):
        if c not in domain:
            domain.append(c)
    range_colors = [COUNTRY_RGB.get(c, stable_rgb(c)) for c in domain]
    return domain, range_colors

def require_columns(df: pd.DataFrame, cols, where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns for {where}: {missing}\n"
            f"Available columns (first 40): {list(df.columns)[:40]}"
        )

def detect_equity_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["eq_tr", "eq_tr_interp", "eq_tr_total", "eq_tr_ann"]
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    return None

def budget_threshold_topk(probs: np.ndarray, budget: float) -> float:
    if len(probs) == 0:
        return 1.0
    return float(np.quantile(probs, 1.0 - budget))

def crisis_episodes(crisis_years: List[int]) -> List[Tuple[int, int]]:
    if not crisis_years:
        return []
    ys = sorted(set(crisis_years))
    eps = []
    s = ys[0]
    p = ys[0]
    for y in ys[1:]:
        if y == p + 1:
            p = y
        else:
            eps.append((s, p))
            s, p = y, y
    eps.append((s, p))
    return eps

def event_level_recall(df_period: pd.DataFrame, alerts: np.ndarray) -> float:
    tmp = df_period[["country", "year", "crisisJST"]].copy()
    tmp["alert"] = alerts
    captured, total = 0, 0
    for _, sub in tmp.groupby("country"):
        sub = sub.sort_values("year")
        crisis_years = sub.loc[sub["crisisJST"] == 1, "year"].astype(int).tolist()
        for cy in crisis_years:
            total += 1
            w = sub[(sub["year"] >= cy - 2) & (sub["year"] <= cy - 1)]
            if w["alert"].sum() > 0:
                captured += 1
    return np.nan if total == 0 else captured / total

def event_table(risk_full: pd.DataFrame, threshold: float, window=(2, 1)) -> pd.DataFrame:
    left, right = window
    rows = []
    for c in sorted(risk_full["country"].unique()):
        d = risk_full[risk_full["country"] == c].sort_values("year").copy()
        crisis_years = d.loc[d["crisisJST"] == 1, "year"].dropna().astype(int).tolist()
        episodes = crisis_episodes(crisis_years)
        for (start, end) in episodes:
            ws, we = start - left, start - right
            w = d[(d["year"] >= ws) & (d["year"] <= we)].copy()
            hit = w[w["prob"] >= threshold].sort_values("year")
            captured = int(not hit.empty)
            first = int(hit["year"].iloc[0]) if captured else np.nan
            lead = int(start - first) if captured else np.nan
            rows.append({
                "country": c,
                "crisis_start": int(start),
                "crisis_end": int(end),
                "warning_window": f"{ws}..{we}",
                "captured": captured,
                "first_warning_year": first,
                "lead_time_years": lead,
            })
    return pd.DataFrame(rows)

def confusion_heatmap(cm: np.ndarray, title: str):
    if ALTAIR_OK:
        d = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).reset_index()
        long = d.melt(id_vars="index", var_name="pred", value_name="count").rename(columns={"index": "actual"})
        chart = alt.Chart(long).mark_rect().encode(
            x=alt.X("pred:N", title=None),
            y=alt.Y("actual:N", title=None),
            color=alt.Color("count:Q"),
            tooltip=["actual:N", "pred:N", "count:Q"],
        ).properties(height=220, title=title)
        text = alt.Chart(long).mark_text().encode(x="pred:N", y="actual:N", text="count:Q")
        st.altair_chart(chart + text, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.imshow(cm)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual 0", "Actual 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title(title)
        st.pyplot(fig, clear_figure=True)

# risk timeline with crisis bands + markers
def altair_risk_with_crisis_bands_and_markers(risk_df, crisis_df, title, threshold=None):
    domain, range_colors = build_color_scale(risk_df["country"].unique())

    base = alt.Chart(risk_df).encode(
        x=alt.X("year:Q", title="Year"),
        y=alt.Y("prob:Q", title="Crisis probability", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("country:N", scale=alt.Scale(domain=domain, range=range_colors), legend=alt.Legend(title="Country")),
        tooltip=["country:N", "year:Q", alt.Tooltip("prob:Q", format=".3f")],
    )

    line = base.mark_line()

    layers = []

    if crisis_df is not None and not crisis_df.empty:
        cdf = crisis_df.copy()
        cdf["x_start"] = cdf["year"] - 0.5
        cdf["x_end"] = cdf["year"] + 0.5

        rect = alt.Chart(cdf).mark_rect(opacity=0.22).encode(
            x="x_start:Q",
            x2="x_end:Q",
            y=alt.value(0),
            y2=alt.value(1),
            color=alt.value("#999999"),
            tooltip=[alt.Tooltip("country:N"), alt.Tooltip("year:Q", title="Crisis year")]
        )
        layers.append(rect)

        m = risk_df.merge(cdf[["country", "year"]], on=["country", "year"], how="inner")
        if not m.empty:
            marker = alt.Chart(m).mark_point(shape="triangle-up", size=90, filled=True, opacity=0.9).encode(
                x="year:Q",
                y="prob:Q",
                color=alt.value("#ff7a00"),  # orange markers
                tooltip=["country:N", "year:Q", alt.Tooltip("prob:Q", format=".3f")]
            )
            layers.append(marker)

    layers.append(line)

    if threshold is not None:
        rule = alt.Chart(pd.DataFrame({"y": [float(threshold)]})).mark_rule(strokeDash=[6, 4], opacity=0.9).encode(y="y:Q")
        layers.append(rule)

    return alt.layer(*layers).properties(height=360, title=title).interactive()

# Matplotlib fallback
def matplotlib_risk_chart_with_crises(risk_df: pd.DataFrame, threshold: float, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Crisis probability")
    ax.set_ylim(0, 1)

    for c in sorted(risk_df["country"].unique()):
        sub = risk_df[risk_df["country"] == c].sort_values("year")
        ax.plot(sub["year"], sub["prob"], label=c)

        crisis_years = sub.loc[sub["crisisJST"] == 1, "year"].astype(int).tolist()
        for y in crisis_years:
            ax.axvspan(y - 0.5, y + 0.5, alpha=0.15)
        crises = sub[sub["crisisJST"] == 1]
        if not crises.empty:
            ax.scatter(crises["year"], crises["prob"], marker="^", s=60)

    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# =============================================================================
# FEATURE ENGINEERING (Leakage-safe)
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    require_columns(df, ["country", "year"], "loading")
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["country"].isin(COUNTRIES_FIXED)].copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    if "crisisJST" not in df.columns:
        df["crisisJST"] = 0
    df["crisisJST"] = df["crisisJST"].fillna(0).astype(int)
    return df

def engineer_macro_features(df: pd.DataFrame):
    df = df.copy()
    required = ["lev", "noncore", "ltd", "hpnom", "cpi", "tloans", "ltrate", "stir", "money", "gdp", "ca"]
    require_columns(df, required, "macro feature engineering")

    df["leverage_risk"] = 1 / (df["lev"].astype(float) + 0.01)

    def expanding_z_causal(s: pd.Series) -> pd.Series:
        mu = s.expanding().mean().shift(1)
        sd = s.expanding().std().shift(1).replace(0, np.nan)
        return (s - mu) / (sd + 1e-9)

    df["noncore_z"] = df.groupby("country")["noncore"].transform(expanding_z_causal)
    df["ltd_z"] = df.groupby("country")["ltd"].transform(expanding_z_causal)
    df["leverage_z"] = df.groupby("country")["leverage_risk"].transform(expanding_z_causal)

    df["banking_fragility"] = 0.4 * df["noncore_z"] + 0.3 * df["ltd_z"] + 0.3 * df["leverage_z"]

    df["hp_real"] = df["hpnom"].astype(float) / (df["cpi"].astype(float) + 1e-9)
    df["hp_trend"] = df.groupby("country")["hp_real"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    df["housing_bubble"] = (df["hp_real"] - df["hp_trend"]) / (df["hp_trend"] + 1e-9)

    df["real_credit"] = df["tloans"].astype(float) / (df["cpi"].astype(float) + 1e-9)
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change()

    df["yield_curve"] = df["ltrate"].astype(float) - df["stir"].astype(float)

    us_ltrate = (
        df[df["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"]
        .to_dict()
    )
    df["us_ltrate"] = df["year"].map(us_ltrate)
    df["sovereign_spread"] = df["ltrate"].astype(float) - df["us_ltrate"].astype(float)

    df["money_gdp"] = df["money"].astype(float) / (df["gdp"].astype(float) + 1e-9)
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()

    df["ca_gdp"] = df["ca"].astype(float) / (df["gdp"].astype(float) + 1e-9)

    macro_features = [
        "housing_bubble",
        "credit_growth",
        "banking_fragility",
        "sovereign_spread",
        "yield_curve",
        "money_expansion",
        "ca_gdp",
    ]
    return df, macro_features

def engineer_behavioral_features(df: pd.DataFrame):
    df = df.copy()
    eq_col = detect_equity_column(df)

    if eq_col:
        df["market_volatility"] = (
            df.groupby("country")[eq_col]
            .transform(lambda s: s.rolling(5, min_periods=3).std().shift(1))
        )
    else:
        df["market_volatility"] = np.nan

    if "risky_tr" in df.columns and "safe_tr" in df.columns:
        df["risk_appetite"] = (df["risky_tr"].astype(float) - df["safe_tr"].astype(float)).groupby(df["country"]).shift(1)
    else:
        df["risk_appetite"] = np.nan

    if "debtgdp" in df.columns and "stir" in df.columns:
        df["debt_service_risk"] = (df["debtgdp"].astype(float) * df["stir"].astype(float)).groupby(df["country"]).shift(1)
    else:
        df["debt_service_risk"] = np.nan

    behav_features = ["market_volatility", "risk_appetite", "debt_service_risk"]
    return df, behav_features

def build_feature_frame(df_raw: pd.DataFrame):
    df, macro = engineer_macro_features(df_raw)
    df, behav = engineer_behavioral_features(df)
    base_features = macro + behav
    keep = ["country", "year", "crisisJST"] + base_features
    df = df[keep].replace([np.inf, -np.inf], np.nan).copy()
    return df, base_features, macro, behav

def apply_causal_cleaning(df: pd.DataFrame, base_features: List[str], train_end_year: int):
    df = df.copy()
    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    for col in base_features:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    df[base_features] = df.groupby("country")[base_features].transform(lambda x: x.ffill(limit=3))

    train_df = df[df["year"] < train_end_year].copy()
    country_medians = train_df.groupby("country")[base_features].median(numeric_only=True).to_dict(orient="index")
    global_medians = train_df[base_features].median(numeric_only=True).to_dict()

    def fill_row(row):
        c = row["country"]
        for f in base_features:
            if pd.isna(row[f]):
                v = country_medians.get(c, {}).get(f, np.nan)
                if pd.isna(v):
                    v = global_medians.get(f, np.nan)
                if pd.isna(v):
                    v = 0.0
                row[f] = v
        return row

    df = df.apply(fill_row, axis=1)
    return df

def create_target(df: pd.DataFrame, horizon: int):
    df = df.copy()
    future_max = None
    for k in range(1, horizon + 1):
        shifted = df.groupby("country")["crisisJST"].shift(-k)
        future_max = shifted if future_max is None else np.maximum(future_max, shifted)
    df["target"] = future_max
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    return df


# =============================================================================
# MODELS
# =============================================================================
def build_model_set():
    return {
        "Logistic Regression": LogisticRegression(max_iter=8000, class_weight="balanced", solver="lbfgs"),
        "Random Forest": RandomForestClassifier(
            n_estimators=600, max_depth=4, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.01, max_depth=3, random_state=42
        ),
        "SVM (RBF, calibrated)": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=2.0, class_weight="balanced"),
            method="sigmoid", cv=3
        ),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2500, random_state=42),
    }


# =============================================================================
# TRAIN/EVAL 
# =============================================================================
def train_eval_budgeted_fixed_model(
    df_target: pd.DataFrame,
    base_features: List[str],
    budget: float,
    model_name: str,
):
    missing_features = [f"{f}_missing" for f in base_features]
    all_features = base_features + missing_features

    train_df = df_target[df_target["year"] < TRAIN_END].copy()
    val_df = df_target[(df_target["year"] >= VAL_START) & (df_target["year"] < VAL_END)].copy()
    test_df = df_target[df_target["year"] >= TEST_START].copy()

    X_train = train_df[all_features]
    y_train = train_df["target"].values
    X_val = val_df[all_features]
    y_val = val_df["target"].values
    X_test = test_df[all_features]
    y_test = test_df["target"].values

    scaler = StandardScaler()
    Xtr_cont = scaler.fit_transform(X_train[base_features])
    Xva_cont = scaler.transform(X_val[base_features])
    Xte_cont = scaler.transform(X_test[base_features])

    Xtr = np.hstack([Xtr_cont, X_train[missing_features].values])
    Xva = np.hstack([Xva_cont, X_val[missing_features].values])
    Xte = np.hstack([Xte_cont, X_test[missing_features].values])

    model = build_model_set()[model_name]
    model.fit(Xtr, y_train)

    val_probs = model.predict_proba(Xva)[:, 1]
    thr_val = budget_threshold_topk(val_probs, budget)

    test_probs = model.predict_proba(Xte)[:, 1]
    test_alerts = (test_probs >= thr_val).astype(int)

    headline = {
        "BestModel": model_name,
        "FrozenThr": float(thr_val),
        "Budget": float(budget),
        "Test_ROC_AUC": float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else np.nan,
        "Test_PR_AUC": float(average_precision_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else np.nan,
        "Test_Precision": float(precision_score(y_test, test_alerts, zero_division=0)),
        "Test_Recall": float(recall_score(y_test, test_alerts, zero_division=0)),
        "Test_F1": float(f1_score(y_test, test_alerts, zero_division=0)),
        "Test_AlertRate": float(test_alerts.mean()),
        "Test_EventRecall": float(event_level_recall(test_df, test_alerts)),
        "Test_Brier": float(brier_score_loss(y_test, test_probs)) if len(np.unique(y_test)) > 1 else np.nan,
    }

    return {
        "headline": headline,
        "splits": (train_df, val_df, test_df),
        "matrices": (Xtr, Xva, Xte, y_train, y_val, y_test),
        "model": model,
        "scaler": scaler,
        "base_features": base_features,
        "missing_features": missing_features,
        "all_features": all_features,
        "test_probs": test_probs,
        "test_alerts": test_alerts,
    }

def validation_model_comparison(df_target: pd.DataFrame, base_features: List[str], budget: float):
    missing_features = [f"{f}_missing" for f in base_features]

    train_df = df_target[df_target["year"] < TRAIN_END].copy()
    val_df = df_target[(df_target["year"] >= VAL_START) & (df_target["year"] < VAL_END)].copy()

    scaler = StandardScaler()
    Xtr_cont = scaler.fit_transform(train_df[base_features])
    Xva_cont = scaler.transform(val_df[base_features])

    Xtr = np.hstack([Xtr_cont, train_df[missing_features].values])
    Xva = np.hstack([Xva_cont, val_df[missing_features].values])

    ytr = train_df["target"].values
    yva = val_df["target"].values

    rows = []
    for name, model in build_model_set().items():
        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xva)[:, 1]
        thr = budget_threshold_topk(probs, budget)
        alerts = (probs >= thr).astype(int)

        rows.append({
            "model": name,
            "ROC-AUC": float(roc_auc_score(yva, probs)) if len(np.unique(yva)) > 1 else np.nan,
            "PR-AUC": float(average_precision_score(yva, probs)) if len(np.unique(yva)) > 1 else np.nan,
            "Val_F1": float(f1_score(yva, alerts, zero_division=0)),
            "Val_AlertRate": float(alerts.mean()),
            "Val_EventRecall": float(event_level_recall(val_df, alerts)),
            "Val_Brier": float(brier_score_loss(yva, probs)) if len(np.unique(yva)) > 1 else np.nan,
            "FrozenThr": float(thr),
        })

    return pd.DataFrame(rows).sort_values(["Val_EventRecall", "PR-AUC", "Val_F1"], ascending=[False, False, False]).reset_index(drop=True)


# =============================================================================
# ROBUSTNESS 
# =============================================================================
def robustness_horizon(df_clean: pd.DataFrame, base_features: List[str], budget: float, model_name: str):
    rows = []
    for h in HORIZON_GRID:
        df_t = create_target(df_clean, horizon=h).reset_index(drop=True)
        out = train_eval_budgeted_fixed_model(df_t, base_features=base_features, budget=budget, model_name=model_name)
        r = out["headline"].copy()
        r["Horizon"] = h
        rows.append(r)
    return pd.DataFrame(rows)

def robustness_budget(df_target: pd.DataFrame, base_features: List[str], horizon: int, model_name: str):
    rows = []
    for b in BUDGET_GRID:
        out = train_eval_budgeted_fixed_model(df_target, base_features=base_features, budget=b, model_name=model_name)
        r = out["headline"].copy()
        r["Budget"] = b
        r["Horizon"] = horizon
        rows.append(r)
    return pd.DataFrame(rows)

# =============================================================================
# ABLATION 
# =============================================================================
def run_ablation(df_target, macro_features, behav_features, budget=0.20):
    results = []
    feature_sets = {
        "Macro only (+missing)": (macro_features, True),
        "Behavioural only (+missing)": (behav_features, True),
        "Macro+Behavioural (+missing)": (macro_features + behav_features, True),
        "Macro+Behavioural (NO missing)": (macro_features + behav_features, False),
    }

    train_df = df_target[df_target["year"] < TRAIN_END].copy()
    val_df = df_target[(df_target["year"] >= VAL_START) & (df_target["year"] < VAL_END)].copy()
    test_df = df_target[df_target["year"] >= TEST_START].copy()

    for name, (feats, use_missing) in feature_sets.items():
        miss = [f"{f}_missing" for f in feats] if use_missing else []

        scaler = StandardScaler()
        Xtr_cont = scaler.fit_transform(train_df[feats])
        Xva_cont = scaler.transform(val_df[feats])
        Xte_cont = scaler.transform(test_df[feats])

        Xtr = Xtr_cont
        Xva = Xva_cont
        Xte = Xte_cont
        if use_missing:
            Xtr = np.hstack([Xtr, train_df[miss].values])
            Xva = np.hstack([Xva, val_df[miss].values])
            Xte = np.hstack([Xte, test_df[miss].values])

        ytr = train_df["target"].values
        yva = val_df["target"].values
        yte = test_df["target"].values

        model = LogisticRegression(max_iter=8000, class_weight="balanced")
        model.fit(Xtr, ytr)

        val_probs = model.predict_proba(Xva)[:, 1]
        thr = budget_threshold_topk(val_probs, budget)

        test_probs = model.predict_proba(Xte)[:, 1]
        test_alerts = (test_probs >= thr).astype(int)

        results.append({
            "Ablation": name,
            "Budget": float(budget),
            "FrozenThr(from VAL)": float(thr),
            "Test_PR_AUC": float(average_precision_score(yte, test_probs)) if len(np.unique(yte)) > 1 else np.nan,
            "Test_EventRecall": float(event_level_recall(test_df, test_alerts)),
            "Test_AlertRate": float(test_alerts.mean()),
            "Test_Precision": float(precision_score(yte, test_alerts, zero_division=0)),
            "Test_Recall": float(recall_score(yte, test_alerts, zero_division=0)),
            "Test_F1": float(f1_score(yte, test_alerts, zero_division=0)),
        })

    return pd.DataFrame(results).sort_values(["Test_EventRecall", "Test_PR_AUC"], ascending=False).reset_index(drop=True)


# =============================================================================
# SHAP 
# =============================================================================
def shap_supported(model_name: str) -> bool:
    return model_name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]

@st.cache_data(show_spinner=False)
def compute_shap_artifacts(
    df_target: pd.DataFrame,
    base_features: List[str],
    model_name: str,
    budget: float,
    sample_n: int = 250,
):
    if not SHAP_OK:
        return None, None, "SHAP is not installed. Add `shap` to requirements.txt."

    if not shap_supported(model_name):
        return None, None, f"SHAP for **{model_name}** is disabled on free-tier (too heavy). Use Logistic Regression / RF / GB."

    missing_features = [f"{f}_missing" for f in base_features]
    all_features = base_features + missing_features

    out = train_eval_budgeted_fixed_model(df_target, base_features=base_features, budget=budget, model_name=model_name)
    model = out["model"]
    scaler = out["scaler"]

    test_df = df_target[df_target["year"] >= TEST_START].copy()
    if len(test_df) == 0:
        test_df = df_target.copy()

    sample_df = test_df.sample(n=min(sample_n, len(test_df)), random_state=42).copy()

    X = sample_df[all_features].copy()
    X_cont = scaler.transform(X[base_features])
    X_mat = np.hstack([X_cont, X[missing_features].values])

    X_sample_df = pd.DataFrame(
        X_mat,
        columns=[f"{f} (z)" for f in base_features] + [f"{f}_missing" for f in base_features]
    )

    try:
        if model_name == "Logistic Regression":
            explainer = shap.LinearExplainer(model, X_sample_df, feature_perturbation="interventional")
            shap_values = explainer(X_sample_df)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample_df)
        return shap_values, X_sample_df, None
    except Exception as e:
        return None, None, f"SHAP failed: {e}"


# =============================================================================
# CORE PIPELINE 
# =============================================================================
@st.cache_data(show_spinner=False)
def build_main_bundle(xlsx_path: str, horizon: int, budget: float, model_name: str):
    df_raw = load_data(xlsx_path)
    df_feat, base_features, macro_features, behav_features = build_feature_frame(df_raw)
    df_clean = apply_causal_cleaning(df_feat, base_features=base_features, train_end_year=TRAIN_END)
    df_target = create_target(df_clean, horizon=horizon).reset_index(drop=True)

    out = train_eval_budgeted_fixed_model(df_target, base_features=base_features, budget=budget, model_name=model_name)

    model = out["model"]
    scaler = out["scaler"]
    all_features = out["all_features"]
    missing_features = out["missing_features"]

    X_full = df_target[all_features].copy()
    X_full_cont = scaler.transform(X_full[base_features])
    X_full_mat = np.hstack([X_full_cont, X_full[missing_features].values])
    probs_full = model.predict_proba(X_full_mat)[:, 1]

    risk_full = df_target[["country", "year", "crisisJST", "target"]].copy()
    risk_full["prob"] = probs_full

    val_results_df = validation_model_comparison(df_target, base_features=base_features, budget=budget)

    return {
        "df_clean": df_clean,
        "df_target": df_target,
        "risk_full": risk_full,
        "base_features": base_features,
        "macro_features": macro_features,
        "behav_features": behav_features,
        "headline": out["headline"],
        "frozen_thr": float(out["headline"]["FrozenThr"]),
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "all_features": all_features,
        "test_probs": out["test_probs"],
        "splits": out["splits"],
        "matrices": out["matrices"],
        "val_results_df": val_results_df,
    }


# =============================================================================
# APP HEADER + SIDEBAR
# =============================================================================
st.title("📉 Financial Crisis Early Warning System (EWS)")
st.caption("A policy-oriented early warning system that identifies rising systemic risk using macro-financial and behavioural indicators.")

if not JST_XLSX.exists():
    st.error(f"Missing dataset file: {JST_XLSX.name}. Put it next to app.py in your Streamlit repo.")
    st.stop()

# Keep model fixed and simple (free-tier safe)
DEFAULT_MODEL = "Logistic Regression"

with st.sidebar:
    st.header("Controls")
    st.markdown("**Countries:** USA + UK (fixed)")

    horizon = st.select_slider("Prediction horizon (t+1..t+H)", options=[1, 2, 3], value=2)
    budget = st.select_slider("Alert budget (on validation)", options=[0.10, 0.20, 0.30], value=0.20)

    st.divider()
    use_manual_threshold = st.checkbox("Override threshold manually", value=False)
    manual_thr = st.slider("Manual threshold", 0.01, 0.99, 0.25, 0.01, disabled=not use_manual_threshold)

    st.divider()
    if st.button("Clear cache & rerun"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Building core model bundle (cached after first run)..."):
    bundle = build_main_bundle(str(JST_XLSX), horizon=horizon, budget=budget, model_name=DEFAULT_MODEL)

df_clean = bundle["df_clean"]
df_target = bundle["df_target"]
risk_full = bundle["risk_full"]
headline = bundle["headline"]
val_results_df = bundle["val_results_df"]

frozen_thr = float(bundle["frozen_thr"])
thr_used = float(manual_thr) if use_manual_threshold else frozen_thr

min_year = int(risk_full["year"].min())
max_year = int(risk_full["year"].max())

colA, colB, colC = st.columns([0.52, 0.23, 0.25])
with colA:
    year_from, year_to = st.slider("Year range (charts)", min_year, max_year, (1900, max_year))
with colB:
    show_crisis_list = st.checkbox("Show crisis years list", value=True)
with colC:
    st.markdown(
        f"""
<div class="kpi-card">
<b>Headline settings</b><br/>
Model: <b>{headline["BestModel"]}</b><br/>
Horizon: <b>t+1..t+{horizon}</b><br/>
Budget: <b>{budget:.0%}</b><br/>
Thr (VAL frozen): <b>{frozen_thr:.3f}</b><br/>
Thr used (charts): <b>{thr_used:.3f}</b>
</div>
""",
        unsafe_allow_html=True,
    )

risk_df = risk_full[(risk_full["year"] >= year_from) & (risk_full["year"] <= year_to)].copy()
crisis_years_df = risk_df[risk_df["crisisJST"] == 1][["country", "year"]].drop_duplicates()

# =============================================================================
# KPI ROW
# =============================================================================
st.markdown("### Headline performance (TEST: 1990+)")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Model", headline["BestModel"])
k2.metric("Event-level recall", f"{headline['Test_EventRecall']:.2f}")
k3.metric("PR-AUC", f"{headline['Test_PR_AUC']:.3f}")
k4.metric("ROC-AUC", f"{headline['Test_ROC_AUC']:.3f}")
k5.metric("Alert rate", f"{headline['Test_AlertRate']:.2f}")
k6.metric("Brier", f"{headline['Test_Brier']:.3f}")

st.caption(
    "Threshold is calibrated on **1970–1990 validation** using the alert budget, then frozen for test. "
    "Dashboard uses a fixed default model for free-tier stability."
)

# =============================================================================
# TABS (exact list requested; no export; no rolling-origin validation)
# =============================================================================
tabs = st.tabs([
    "📈 Risk Timeline",
    "📊 Model Comparison",
    "🧪 Robustness",
    "🧪 Robustness (Run)",
    "🧩 Ablation (Run)",
    "📌 Policy Evidence",
    "SHAP",
])

# -----------------------------------------------------------------------------
# TAB 1: Risk Timeline
with tabs[0]:
    st.subheader("Crisis risk over time (triangles = crisis markers)")

    if risk_df.empty:
        st.warning("No data in this selection.")
    else:
        if ALTAIR_OK:
            st.altair_chart(
                altair_risk_with_crisis_bands_and_markers(
                    risk_df.sort_values(["country", "year"]),
                    crisis_years_df,
                    title="Predicted crisis probability (with crisis-year bands + markers)",
                    threshold=thr_used
                ),
                use_container_width=True
            )
        else:
            matplotlib_risk_chart_with_crises(
                risk_df.sort_values(["country", "year"]),
                threshold=thr_used,
                title="Predicted crisis probability (matplotlib fallback)"
            )
            st.info("Install **altair** for interactive hover + clean crisis-year bands (recommended).")

        st.markdown(
            '<div class="small-note">Interpretation: pre-crisis risk rises are desirable. Risk spikes far from crises are false alarms.</div>',
            unsafe_allow_html=True
        )

        if show_crisis_list:
            with st.expander("Crisis years in selected window"):
                for c in sorted(risk_df["country"].unique()):
                    years = crisis_years_df[crisis_years_df["country"] == c]["year"].astype(int).tolist()
                    st.write(f"**{c}:** " + (", ".join(map(str, years)) if years else "none"))

# -----------------------------------------------------------------------------
# TAB 2: Model Comparison
with tabs[1]:
    st.subheader("Fixed validation comparison (1970–1990)")
    st.dataframe(val_results_df, use_container_width=True)

    train_df, val_df, test_df = bundle["splits"]
    Xtr, Xva, Xte, ytr, yva, yte = bundle["matrices"]
    test_probs = bundle["test_probs"]

    st.subheader("Test performance at threshold (TEST: 1990+)")
    test_alerts_thr = (test_probs >= thr_used).astype(int)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PR-AUC", f"{average_precision_score(yte, test_probs):.3f}" if len(np.unique(yte)) > 1 else "n/a")
    c2.metric("ROC-AUC", f"{roc_auc_score(yte, test_probs):.3f}" if len(np.unique(yte)) > 1 else "n/a")
    c3.metric("Precision", f"{precision_score(yte, test_alerts_thr, zero_division=0):.3f}")
    c4.metric("Recall", f"{recall_score(yte, test_alerts_thr, zero_division=0):.3f}")
    c5.metric("F1", f"{f1_score(yte, test_alerts_thr, zero_division=0):.3f}")

    cm = confusion_matrix(yte, test_alerts_thr)
    confusion_heatmap(cm, title="Confusion matrix (TEST)")

# -----------------------------------------------------------------------------
# TAB 3: Robustness (explain only)
with tabs[2]:
    st.subheader("Robustness (what we test and why)")
    st.markdown(
        """
This dissertation uses robustness checks to show results are not a “one-off”.

**Two key sensitivities:**
1) **Horizon sensitivity (H=1,2,3):** does the model still warn early if we change how far ahead we predict?
2) **Budget sensitivity (10%, 20%, 30%):** does performance stay reasonable if policymakers tolerate fewer/more alerts?

Use the next tab (**Robustness (Run)**) to compute the tables on-demand (free-tier safe).
"""
    )

# -----------------------------------------------------------------------------
# TAB 4: Robustness (Run)
with tabs[3]:
    st.subheader("Robustness checks (run on-demand)")
    st.caption("Runs on demand to keep Streamlit Cloud free-tier fast.")

    if st.button("Run robustness: Horizon sensitivity (H=1,2,3)"):
        with st.spinner("Running horizon robustness..."):
            df_clean_local = df_clean.copy()
            base_features_local = bundle["base_features"]
            horizon_sens = robustness_horizon(df_clean_local, base_features_local, budget=budget, model_name=DEFAULT_MODEL)

        show_cols = ["BestModel", "Budget", "FrozenThr", "Test_PR_AUC", "Test_EventRecall", "Test_AlertRate", "Test_F1", "Test_ROC_AUC", "Horizon"]
        st.dataframe(horizon_sens[show_cols], use_container_width=True)

    if st.button("Run robustness: Budget sensitivity (10%,20%,30%)"):
        with st.spinner("Running budget robustness..."):
            base_features_local = bundle["base_features"]
            budget_sens = robustness_budget(df_target, base_features_local, horizon=horizon, model_name=DEFAULT_MODEL)

        show_cols2 = ["BestModel", "Budget", "FrozenThr", "Test_PR_AUC", "Test_EventRecall", "Test_AlertRate", "Test_F1", "Test_ROC_AUC", "Horizon"]
        st.dataframe(budget_sens[show_cols2], use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: Ablation (Run)
with tabs[4]:
    st.subheader("Ablation study (macro vs behavioural vs combined)")
    st.caption("Runs on demand (cached after first run).")

    if st.button("Run ablation"):
        with st.spinner("Running ablation..."):
            ab = run_ablation(df_target, bundle["macro_features"], bundle["behav_features"], budget=budget)
        st.dataframe(ab, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 6: Policy Evidence
with tabs[5]:
    st.subheader("Policy evidence: event capture and lead time")

    wl = st.slider("Warning window start (years before crisis)", 1, 5, 2, 1)
    wr = st.slider("Warning window end (years before crisis)", 1, 3, 1, 1)

    if wl <= wr:
        st.warning("Window must be like t-2..t-1 (start > end). Increase the start slider.")
    else:
        ev = event_table(risk_full, threshold=thr_used, window=(wl, wr))
        if ev.empty:
            st.info("No crisis episodes detected in this selection.")
        else:
            st.dataframe(ev.sort_values(["country", "crisis_start"]), use_container_width=True)
            summary = ev.groupby("country")["captured"].agg(["sum", "count"]).reset_index()
            summary["capture_rate"] = summary["sum"] / summary["count"]
            st.markdown("**Capture rate by country**")
            st.dataframe(summary, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 7: SHAP
with tabs[6]:
    st.subheader("SHAP explainability (what features drive the risk score)")
    st.caption("Runs on-demand. Best on Logistic Regression (fast + interpretable).")

    if not SHAP_OK:
        st.error("SHAP is not installed. Add `shap` to requirements.txt and redeploy.")
    else:
        st.markdown(f"**Current model:** `{DEFAULT_MODEL}`")

        sample_n = st.slider("SHAP sample size (higher = slower)", 50, 600, 250, 50)

        if st.button("Run SHAP summary"):
            with st.spinner("Computing SHAP (cached after first run)..."):
                shap_values, X_sample_df, err = compute_shap_artifacts(
                    df_target=df_target,
                    base_features=bundle["base_features"],
                    model_name=DEFAULT_MODEL,
                    budget=budget,
                    sample_n=sample_n,
                )

            if err is not None:
                st.warning(err)
            else:
                st.success("SHAP computed.")

                st.markdown("**Global importance (mean |SHAP|):**")
                fig = plt.figure(figsize=(9, 4))
                shap.plots.bar(shap_values, max_display=18, show=False)
                st.pyplot(fig, clear_figure=True)

                st.markdown("**Summary plot (direction + strength):**")
                fig2 = plt.figure(figsize=(9, 4))
                shap.summary_plot(shap_values.values, X_sample_df, show=False, max_display=18)
                st.pyplot(fig2, clear_figure=True)

                st.markdown(
                    '<div class="small-note">Positive SHAP pushes crisis probability up; negative SHAP pushes it down.</div>',
                    unsafe_allow_html=True
                )
