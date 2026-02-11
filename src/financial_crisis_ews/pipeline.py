# src/financial_crisis_ews/pipeline.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .data import load_raw_jst  # ✅ your data.py currently defines load_raw_jst
from .features import make_time_safe_features
from .model import fit_model, predict_proba
from .evaluate import compute_metrics, rolling_origin_evaluation


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _infer_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _ensure_year_int(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df.copy()
    y = out[year_col]

    if np.issubdtype(y.dtype, np.number):
        out[year_col] = pd.to_numeric(y, errors="coerce").astype("Int64")
        return out

    y_dt = pd.to_datetime(y, errors="coerce")
    if y_dt.notna().any():
        out[year_col] = y_dt.dt.year.astype("Int64")
        return out

    out[year_col] = pd.to_numeric(y, errors="coerce").astype("Int64")
    return out


def _resolve_core_columns(
    df: pd.DataFrame,
    target_col: str,
    year_col: Optional[str],
    country_col: Optional[str],
) -> Tuple[str, str, str]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    inferred_year = year_col or _infer_col(df, ["year", "time", "date", "yr"])
    if inferred_year is None:
        raise KeyError("Could not find a year-like column. Provide --year-col.")

    inferred_country = country_col or _infer_col(df, ["country", "iso", "iso3", "cty"])
    if inferred_country is None:
        raise KeyError("Could not find a country-like column. Provide --country-col.")

    return inferred_country, inferred_year, target_col


def run_pipeline(
    raw_file: str,
    target_col: str = "crisisJST",
    budget: float = 0.20,
    year_col: Optional[str] = None,
    country_col: Optional[str] = None,
    out_dir: str = "models",
    reports_dir: str = "reports",
    do_rolling: bool = True,
    n_folds: int = 5,
) -> dict:
    raw_path = Path(raw_file)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_raw_jst(raw_path)

    country_col, year_col, target_col = _resolve_core_columns(df, target_col, year_col, country_col)
    df = _ensure_year_int(df, year_col)

    if df[year_col].isna().all():
        raise ValueError(f"Year column '{year_col}' has no usable values after parsing.")

    df = df.rename(columns={country_col: "country", year_col: "year"}).copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    Xy = make_time_safe_features(df, target_col=target_col)
    X = Xy.drop(columns=[target_col])
    y = Xy[target_col].fillna(0).astype(int).values

    model, preprocessor = fit_model(X, y)
    probs = predict_proba(model, preprocessor, X)

    metrics = compute_metrics(y_true=y, y_score=probs, budget=budget)

    out_dir_p = Path(out_dir)
    reports_dir_p = Path(reports_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    reports_dir_p.mkdir(parents=True, exist_ok=True)

    model_path = out_dir_p / "model.joblib"
    preproc_path = out_dir_p / "preprocessor.joblib"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preproc_path)

    artifacts = {
        "model_path": str(model_path.resolve()),
        "preprocessor_path": str(preproc_path.resolve()),
        "metrics": metrics,
        "used_columns": {"target_col": target_col, "country_col": "country", "year_col": "year"},
    }

    if do_rolling:
        rolling_csv = reports_dir_p / "rolling_metrics.csv"
        rolling = rolling_origin_evaluation(
            df=df,
            target_col=target_col,
            budget=budget,
            n_folds=n_folds,
            reports_csv=str(rolling_csv),
        )
        artifacts["rolling"] = rolling
        artifacts["reports_csv"] = str(rolling_csv.resolve())

    return artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-file", required=True, help="Path to the JST Excel file (e.g. data/raw/JSTdataset.xlsx)")
    p.add_argument("--target-col", default="crisisJST")
    p.add_argument("--budget", type=float, default=0.20)
    p.add_argument("--year-col", default=None)
    p.add_argument("--country-col", default=None)
    p.add_argument("--out-dir", default="models")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--no-rolling", action="store_true")
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()

    artifacts = run_pipeline(
        raw_file=args.raw_file,
        target_col=args.target_col,
        budget=args.budget,
        year_col=args.year_col,
        country_col=args.country_col,
        out_dir=args.out_dir,
        reports_dir=args.reports_dir,
        do_rolling=(not args.no_rolling),
        n_folds=args.n_folds,
    )

    print("\nDONE.")
    # ✅ print exactly the “old style” key metrics on console
    if "metrics" in artifacts:
        print("Metrics:")
        for k, v in artifacts["metrics"].items():
            print(f"{k}: {v}")
    if "reports_csv" in artifacts:
        print(f"reports_csv: {artifacts['reports_csv']}")
    print(f"model_path: {artifacts['model_path']}")
    print(f"preprocessor_path: {artifacts['preprocessor_path']}")


if __name__ == "__main__":
    main()
