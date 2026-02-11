import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load JST Excel file."""
    df = pd.read_excel(path)
    return df

def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

