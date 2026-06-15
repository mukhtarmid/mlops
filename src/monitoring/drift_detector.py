from pathlib import Path
from typing import Dict, Any
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILE = PROJECT_ROOT / "data" / "reference" / "train.csv"
CURRENT_FILE = PROJECT_ROOT / "data" / "production" / "live_data.csv"


def calculate_drift() -> Dict[str, Any]:
    """Uses Evidently AI to evaluate structural and feature value drift between training and inference data."""
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference baseline data missing at {REFERENCE_FILE}. Training pipeline must save it."
        )
    if not CURRENT_FILE.exists():
        raise FileNotFoundError(
            f"Production execution data missing at {CURRENT_FILE}. Run Phase 2 store update first."
        )

    # Load baseline split vs current production tracking
    reference_df = pd.read_csv(REFERENCE_FILE)
    current_df = pd.read_csv(CURRENT_FILE)

    # Exclude prediction targets if you only want feature drift analysis
    # report = Report(metrics=[DataDriftPreset(columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    result_dict = report.as_dict()

    # Extract global summary status flags
    dataset_drift = result_dict["metrics"][0]["result"]["dataset_drift"]

    return {"dataset_drift": bool(dataset_drift)}


if __name__ == "__main__":
    try:
        print(calculate_drift())
    except Exception as e:
        print(f"Drift Pipeline Error: {e}")