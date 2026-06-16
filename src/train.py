from pathlib import Path
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.config import MLFLOW_TRACKING_URI

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris-training")

def promote_if_better(
    client,
    candidate_version,
    candidate_accuracy
):
    try:
        production_version = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                "production"
            )
        )

        production_accuracy = float(
            production_version.tags.get(
                "accuracy",
                0.0
            )
        )

    except Exception:
        # No production model exists yet
        production_version = None
        production_accuracy = 0.0

    if candidate_accuracy > production_accuracy:

        client.set_registered_model_alias(
            MODEL_NAME,
            "production",
            candidate_version
        )

        print(
            f"Promoted v{candidate_version} "
            f"to production "
            f"({candidate_accuracy:.4f} > "
            f"{production_accuracy:.4f})"
        )

    else:

        print(
            f"Kept current production "
            f"({production_accuracy:.4f} >= "
            f"{candidate_accuracy:.4f})"
        )

MODEL_NAME = "iris_classifier"


def load_data():

    iris = load_iris()

    X = iris.data
    y = iris.target

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

def save_reference_dataset(X_train):

    reference_dir = Path(
        "data/reference"
    )

    reference_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    reference_df = pd.DataFrame(
        X_train,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    )

    reference_df.to_csv(
        reference_dir / "train.csv",
        index=False
    )

def train_model(
    X_train,
    y_train,
    n_estimators=200
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return accuracy


def register_model(
    model,
    accuracy,
    n_estimators
):
    client = MlflowClient()

    with mlflow.start_run():

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "dataset",
            "iris"
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        latest_version = max(
            versions,
            key=lambda x: int(x.version)
        )

        latest_version_number = latest_version.version

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=latest_version_number,
            key="accuracy",
            value=str(accuracy)
        )

        # Always assign new model to staging
        client.set_registered_model_alias(
            MODEL_NAME,
            "staging",
            latest_version_number
        )

        promote_if_better(
            client,
            latest_version_number,
            accuracy
        )

        return latest_version_number

def run_training_pipeline(
    n_estimators=200
):

    setup_mlflow()
    
    X_train, X_test, y_train, y_test = load_data()

    save_reference_dataset(
        X_train
    )

    model = train_model(
        X_train,
        y_train,
        n_estimators
    )

    accuracy = evaluate_model(
        model,
        X_test,
        y_test
    )

    if accuracy < 0.90:
        raise Exception(
            f"Accuracy too low: {accuracy}"
        )

    register_model(
        model,
        accuracy,
        n_estimators
    )

    return accuracy


if __name__ == "__main__":

    accuracy = run_training_pipeline()

    print(
        f"Training completed. Accuracy={accuracy}"
    )