from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    # repo root = .../financial-crisis-ews
    root: Path = Path(__file__).resolve().parents[2]
    data_raw: Path = root / "data" / "raw"
    data_processed: Path = root / "data" / "processed"
    models: Path = root / "models"
    reports: Path = root / "reports"

PATHS = Paths()

