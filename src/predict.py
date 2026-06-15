import joblib
import mlflow
import pandas as pd

# model = None
LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

model = mlflow.pyfunc.load_model(
    model_uri="models:/iris_classifier@production"
)

# def load_model():

#     global model

#     if model is None:
#         model = joblib.load("models/model.pkl")

#     return model


def predict(features):

    df = pd.DataFrame(
        [features],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    prediction = model.predict(df)

    prediction = prediction[0]

    prediction = int(prediction)

    return LABELS[prediction]