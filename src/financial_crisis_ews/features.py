from __future__ import annotations

import numpy as np
import pandas as pd

CRISIS_COL_DEFAULT = "crisisJST"

def detect_equity_column(df: pd.DataFrame) -> str | None:
    """
    Picks the first equity-related column found, matching your notebook intent.
    Adjust priority here if needed.
    """
    candidates = [
        "eq_tr", "eq_tr_interp", "eq_capgain", "eq_capgain_interp", "eq_dp", "eq_dp_interp"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def engineer_macro_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df_raw.copy()

    # Housing bubble
    df["housing_bubble"] = df["hp"].astype(float) / (df["cpi"].astype(float) + 1e-9)

    # Credit growth
    df["credit_growth"] = df.groupby("country")["real_credit"].pct_change(fill_method=None)

    # Banking fragility proxy
    df["banking_fragility"] = df["debtgdp"].astype(float) / (df["gdp"].astype(float) + 1e-9)

    # Yield curve
    df["yield_curve"] = df["ltrate"].astype(float) - df["stir"].astype(float)

    # Sovereign spread relative to USA long rate (year-mapped)
    us_ltrate = (
        df[df["country"] == "USA"]
        .drop_duplicates("year")
        .set_index("year")["ltrate"]
        .to_dict()
    )
    df["us_ltrate"] = df["year"].map(us_ltrate)
    df["sovereign_spread"] = df["ltrate"].astype(float) - df["us_ltrate"].astype(float)

    # Money expansion
    df["money_gdp"] = df["money"].astype(float) / (df["gdp"].astype(float) + 1e-9)
    df["money_expansion"] = df.groupby("country")["money_gdp"].pct_change()

    # Current account / GDP
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

def engineer_behavioral_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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
        df["risk_appetite"] = (
            (df["risky_tr"].astype(float) - df["safe_tr"].astype(float))
            .groupby(df["country"]).shift(1)
        )
    else:
        df["risk_appetite"] = np.nan

    if "debtgdp" in df.columns and "stir" in df.columns:
        df["debt_service_risk"] = (
            (df["debtgdp"].astype(float) * df["stir"].astype(float))
            .groupby(df["country"]).shift(1)
        )
    else:
        df["debt_service_risk"] = np.nan

    behav_features = ["market_volatility", "risk_appetite", "debt_service_risk"]
    return df, behav_features

def build_feature_frame(
    df_raw: pd.DataFrame,
    crisis_col: str = CRISIS_COL_DEFAULT
) -> tuple[pd.DataFrame, list[str]]:
    df, macro = engineer_macro_features(df_raw)
    df, behav = engineer_behavioral_features(df)

    base_features = macro + behav
    keep_cols = ["country", "year", crisis_col] + base_features
    df = df[keep_cols].replace([np.inf, -np.inf], np.nan).copy()

    return df, base_features

def apply_causal_cleaning(
    df: pd.DataFrame,
    base_features: list[str],
    train_end_year: int
) -> pd.DataFrame:
    df = df.copy()

    # War-year exclusions (same as notebook)
    df = df[~df["year"].between(1914, 1918)]
    df = df[~df["year"].between(1939, 1945)]

    # Missing flags
    missing_flags = {f"{col}_missing": df[col].isna().astype(int) for col in base_features}
    df = pd.concat([df, pd.DataFrame(missing_flags, index=df.index)], axis=1)

    # Forward-fill only within country, limited
    df[base_features] = df.groupby("country")[base_features].transform(lambda x: x.ffill(limit=3))

    # Train-only medians
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

    return df.apply(fill_row, axis=1)

def create_target(
    df: pd.DataFrame,
    crisis_col: str = CRISIS_COL_DEFAULT,
    horizon: int = 2
) -> pd.DataFrame:
    """
    Creates a pre-crisis target: 1 if crisis occurs in next 1..horizon years (within country).
    """
    df = df.sort_values(["country", "year"]).copy()

    future_flags = []
    for h in range(1, horizon + 1):
        future_flags.append(df.groupby("country")[crisis_col].shift(-h).fillna(0).astype(int))

    df["target"] = (pd.concat(future_flags, axis=1).max(axis=1)).astype(int)
    return df

