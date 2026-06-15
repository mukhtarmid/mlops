import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_ROOT / "logs" / "predictions.jsonl"
STORE_FILE = PROJECT_ROOT / "data" / "production" / "live_data.csv"


def update_store() -> None:
    """Transforms raw JSONL raw logs into an aligned production dataset for drift analytics."""
    if not LOG_FILE.exists():
        print(f"No prediction logs found at {LOG_FILE}. Skipping store update.")
        return

    STORE_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Read and parse JSONL records safely
    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("Prediction log file is empty.")
        return

    # Process and flatten the list representation into independent features
    flattened_data = []
    for record in records:
        features = record["features"]
        flattened_data.append(
            {
                "sepal_length": features[0],
                "sepal_width": features[1],
                "petal_length": features[2],
                "petal_width": features[3],
                "prediction": record["prediction"],
            }
        )

    # Overwrite/Create production dataset
    df_live = pd.DataFrame(flattened_data)
    df_live.to_csv(STORE_FILE, index=False)
    print(f"Successfully processed {len(df_live)} rows into {STORE_FILE}")


if __name__ == "__main__":
    update_store()