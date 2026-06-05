from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():
    data = load_iris()

    X = data.data
    y = data.target

    model = RandomForestClassifier()

    model.fit(X, y)

    joblib.dump(model, "models/model.pkl")

    print("Model saved")


if __name__ == "__main__":
    main()