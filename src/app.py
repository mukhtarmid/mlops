from fastapi import FastAPI
from pydantic import BaseModel
from src.logger import logger
from src.predict import predict
from src.monitoring.prediction_logger import log_prediction

app = FastAPI()

class IrisFeatures(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():

    return {
        "status": "healthy"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

@app.get("/model-info")
def model_info():

    return {
        "model_name": "iris_classifier",
        "stage": "Production"
    }

@app.post("/predict")
def get_prediction(data: IrisFeatures):

    global request_count

    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]
    logger.info(
        f"Prediction request received: {features}"
    )

    request_count += 1

    prediction = predict(features)
    if prediction in prediction_counter:
        prediction_counter[prediction] += 1
    
    log_prediction(
        features,
        prediction
    )

    logger.info(
        f"Prediction result: {prediction}"
    )

    return {
        "prediction": prediction
    }

request_count = 0
prediction_counter = {
    "setosa": 0,
    "versicolor": 0,
    "virginica": 0
}

@app.get("/metrics")
def metrics():

    return {
        "requests": request_count,
        "predictions": prediction_counter
    }