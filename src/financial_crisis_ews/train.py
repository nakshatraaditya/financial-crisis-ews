from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from financial_crisis_ews.io import load_data, require_columns
from financial_crisis_ews.features import (
    build_feature_frame, apply_causal_cleaning, create_target
)
from financial_crisis_ews.evaluation import (
    budget_threshold_topk, event_level_recall, compute_prob_metrics
)

def rolling_train_eval(
    df_target: pd.DataFrame,
    base_features: list[str],
    crisis_col: str,
    budget: float,
    step_years: int = 10,
    min_train_years: int = 30,
) -> pd.DataFrame:
    df_target = df_target.sort_values(["country", "year"]).copy()
    years = sorted(df_target["year"].unique())

    missing_features = [f"{f}_missing" for f in base_features if f"{f}_missing" in df_target.columns]
    all_features = base_features + missing_features

    rows = []

    for i in range(0, len(years), step_years):
        train_end_year = int(years[i])
        if train_end_year < int(years[0]) + min_train_years:
            continue

        test_end_year = train_end_year + step_years

        train_df = df_target[df_target["year"] < train_end_year].copy()
        test_df  = df_target[(df_target["year"] >= train_end_year) & (df_target["year"] < test_end_year)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue
        if train_df["target"].nunique() < 2:
            # no positives in training -> skip
            continue

        scaler = StandardScaler()
        Xtr_cont = scaler.fit_transform(train_df[base_features])
        Xte_cont = scaler.transform(test_df[base_features])

        if missing_features:
            Xtr = np.hstack([Xtr_cont, train_df[missing_features].values])
            Xte = np.hstack([Xte_cont, test_df[missing_features].values])
        else:
            Xtr, Xte = Xtr_cont, Xte_cont

        ytr = train_df["target"].values.astype(int)
        yte = test_df["target"].values.astype(int)

        model = LogisticRegression(max_iter=8000, class_weight="balanced", solver="lbfgs")
        model.fit(Xtr, ytr)

        # Threshold calibrated on this test window to hit budget (matches your pipeline behaviour)
        probs = model.predict_proba(Xte)[:, 1]
        thr = budget_threshold_topk(probs, budget)
        alerts = (probs >= thr).astype(int)

        base_rate = float(yte.mean())
        prob_metrics = compute_prob_metrics(yte, probs)

        rows.append({
            "cutoff_year": train_end_year,
            "test_end_year": int(test_end_year),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "base_rate": base_rate,
            "pr_auc": prob_metrics["pr_auc"] if prob_metrics["pr_auc"] == prob_metrics["pr_auc"] else -0.0,  # keep your csv look
            "brier": prob_metrics["brier"] if prob_metrics["brier"] == prob_metrics["brier"] else np.nan,
            "alert_threshold": float(thr),
            "alert_rate": float(alerts.mean()),
            "crisis_recall": float(event_level_recall(test_df, alerts, crisis_col=crisis_col)) if (test_df[crisis_col] == 1).any() else 0.0,
            "onset_recall": np.nan,  # keep field for compatibility (you can wire onset later)
            "tp": int(((alerts == 1) & (yte == 1)).sum()),
            "fp": int(((alerts == 1) & (yte == 0)).sum()),
            "tn": int(((alerts == 0) & (yte == 0)).sum()),
            "fn": int(((alerts == 0) & (yte == 1)).sum()),
        })

    return pd.DataFrame(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-file", required=True, help="Path to JSTdatasetR6.xlsx")
    p.add_argument("--target-col", default="crisisJST", help="Crisis column name (e.g., crisisJST)")
    p.add_argument("--budget", type=float, default=0.20, help="Alert budget (e.g., 0.20 = 20%)")
    p.add_argument("--train-end", type=int, default=1950, help="Train-end year used for causal cleaning medians")
    p.add_argument("--horizon", type=int, default=2, help="Warning horizon (years ahead)")
    p.add_argument("--step-years", type=int, default=10, help="Rolling window step size")
    p.add_argument("--min-train-years", type=int, default=30, help="Minimum training span (years)")
    p.add_argument("--reports-dir", default="reports", help="Where to write outputs")
    args = p.parse_args()

    df_raw = load_data(args.raw_file)

    # minimal required columns for your engineered features
    require_columns(df_raw, [
        "country", "year", args.target_col,
        "hp", "cpi", "real_credit", "debtgdp", "gdp",
        "ltrate", "stir", "money", "ca"
    ])

    df_feat, base_features = build_feature_frame(df_raw, crisis_col=args.target_col)
    df_clean = apply_causal_cleaning(df_feat, base_features, train_end_year=args.train_end)
    df_target = create_target(df_clean, crisis_col=args.target_col, horizon=args.horizon).reset_index(drop=True)

    out_df = rolling_train_eval(
        df_target=df_target,
        base_features=base_features,
        crisis_col=args.target_col,
        budget=args.budget,
        step_years=args.step_years,
        min_train_years=args.min_train_years,
    )

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "rolling_metrics.csv"
    out_df.to_csv(out_path, index=False)

    # Print a quick summary like your CLI output
    pr_auc_mean = float(out_df["pr_auc"].replace(-0.0, np.nan).mean())
    pr_auc_median = float(out_df["pr_auc"].replace(-0.0, np.nan).median())
    brier_mean = float(out_df["brier"].mean())

    print("\nDONE. Metrics:")
    print(f"folds: {len(out_df)}")
    print(f"pr_auc_mean: {pr_auc_mean}")
    print(f"pr_auc_median: {pr_auc_median}")
    print(f"brier_mean: {brier_mean}")
    print(f"crisis_recall_mean: {float(out_df['crisis_recall'].mean()) if len(out_df) else float('nan')}")
    print(f"base_rate_mean: {float(out_df['base_rate'].mean()) if len(out_df) else float('nan')}")
    print(f"alert_rate_mean: {float(out_df['alert_rate'].mean()) if len(out_df) else float('nan')}")
    print(f"reports_csv: {out_path.resolve()}")

if __name__ == "__main__":
    main()

