# tests/test_evaluate.py (MLflow-enabled, version-only)
import os
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pyfunc

# Data path (DVC tracks the dataset as before)
DATA_PATH = "data/raw/iris.csv"

@pytest.fixture(scope="module")
def model():
    # MLflow tracking URI
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set to run tests that pull model from MLflow registry.")
    mlflow.set_tracking_uri(mlflow_uri)

    # Model name and version
    model_name = os.environ.get("TEST_MODEL_NAME", "iris-classifier")
    model_version = os.environ.get("TEST_MODEL_VERSION", "latest")  # only version, no stage
    model_uri = f"models:/{model_name}/{model_version}"

    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    assert hasattr(model, "predict")
    return model

def test_data_exists():
    """Check that dataset exists and has expected structure."""
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} does not exist"
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset is empty"
    assert "species" in df.columns, "Target column 'species' missing"

def test_model_predicts_correctly(model):
    """Check model predictions on the dataset and basic accuracy."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["species"])
    y = df["species"]

    preds = model.predict(X)
    # Normalize to numpy array length if necessary
    assert len(preds) == len(y), "Prediction length mismatch"
    acc = accuracy_score(y, preds)
    assert acc > 0.8, f"Sanity check failed: accuracy too low ({acc:.2f})"

    # Save accuracy to a file for CML
    with open("result.log", "w") as f:
        f.write(f"Sanity Test Accuracy: {acc:.4f}\n")
