from pathlib import Path
import pandas as pd

def load_raw_jst(path: Path) -> pd.DataFrame:
    """
    Load the JST Excel file.
    If your file needs a specific sheet_name, add it here.
    """
    return pd.read_excel(path)

def save_processed(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

