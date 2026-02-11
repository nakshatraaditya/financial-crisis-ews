from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .data import load_raw_excel
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
    for c in df.columns:
        if _norm(c) in {_norm(x) for x in candidates}:
            return c
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

    inferred_year = year_col or _infer_col(df, ["year", "Year", "YEAR", "time", "date", "yr"])
    if inferred_year is None:
        raise KeyError("Could not find a year-like column. Provide --year-col.")

    inferred_country = country_col or _infer_col(df, ["country", "Country", "COUNTRY", "iso", "iso3", "cty"])
    if inferred_country is None:
        raise KeyError("Could not find a country-like column. Provide --country-col.")

    return inferred_country, inferred_year, target_col


def run_pipeline(
    raw_file: str,
    target_col: str = "crisis",
    budget: float = 0.20,
    year_col: Optional[str] = None,
    country_col: Optional[str] = None,
    out_dir: str = "models",
    reports_dir: str = "reports",
    do_rolling: bool = True,
    n_folds: int = 5,
) -> dict:
    df = load_raw_excel(raw_file)

    country_col, year_col, target_col = _resolve_core_columns(df, target_col, year_col, country_col)
    df = _ensure_year_int(df, year_col)

    if df[year_col].isna().all():
        raise ValueError(f"Year column '{year_col}' has no usable values after parsing.")

    df = df.rename(columns={country_col: "country", year_col: "year"}).copy()

    if "country" not in df.columns or "year" not in df.columns:
        raise KeyError("Failed to standardize columns to 'country' and 'year'.")

    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    Xy = make_time_safe_features(df, target_col=target_col)
    X = Xy.drop(columns=[target_col])
    y = Xy[target_col].astype(int).values

    model, preprocessor = fit_model(X, y)
    probs = predict_proba(model, preprocessor, X)

    metrics = compute_metrics(y_true=y, y_score=probs, budget=budget)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    artifacts = {
        "model_path": str(Path(out_dir) / "model.joblib"),
        "preprocessor_path": str(Path(out_dir) / "preprocessor.joblib"),
        "metrics": metrics,
        "used_columns": {"target_col": target_col, "country_col": "country", "year_col": "year"},
    }

    if do_rolling:
        rolling = rolling_origin_evaluation(
            df=df,
            target_col=target_col,
            budget=budget,
            n_folds=n_folds,
            reports_csv=str(Path(reports_dir) / "rolling_metrics.csv"),
        )
        artifacts["rolling"] = rolling

    return artifacts

