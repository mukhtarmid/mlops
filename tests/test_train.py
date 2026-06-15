from src.train import train_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def test_model_training():

    model = train_model()

    assert model is not None

def test_model_type():

    model = train_model()

    assert isinstance(
        model,
        RandomForestClassifier
    )

def test_model_is_fitted():

    model = train_model()

    assert hasattr(
        model,
        "feature_importances_"
    )

def test_model_accuracy():

    model = train_model()

    data = load_iris()

    score = model.score(
        data.data,
        data.target
    )

    assert score > 0.90