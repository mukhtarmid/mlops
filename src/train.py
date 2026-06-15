from pathlib import Path
import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import joblib

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    "iris-training"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pkl"

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


def save_model(model):

    joblib.dump(
        model,
        MODEL_PATH
    )

    return str(MODEL_PATH)


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

        model_info = mlflow.sklearn.log_model(
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

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=latest_version.version,
            key="accuracy",
            value=str(accuracy)
        )

        return latest_version.version

def run_training_pipeline(
    n_estimators=200
):

    X_train, X_test, y_train, y_test = load_data()

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

    save_model(model)

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