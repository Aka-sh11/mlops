# tests/test_evaluate.py
import os
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

# Paths to DVC-tracked files
DATA_PATH = "data/raw/iris.csv"
MODEL_PATH = "tmp/model.joblib"

# Fixture to load model once per module
@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} does not exist"
    m = joblib.load(MODEL_PATH)
    assert hasattr(m, "predict")
    return m

def test_data_exists():
    """Check that dataset exists and has expected structure."""
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} does not exist"
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Dataset is empty"
    assert "species" in df.columns, "Target column 'species' missing"

def test_model_exists_and_loads(model):
    """Check that model file exists and can be loaded (handled by fixture)."""
    pass  # model fixture already asserts existence and predict method

def test_model_predicts_correctly(model):
    """Check model predictions on the dataset and basic accuracy."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["species"])
    y = df["species"]

    preds = model.predict(X)
    assert len(preds) == len(y), "Prediction length mismatch"
    acc = accuracy_score(y, preds)
    assert acc > 0.7, f"Sanity check failed: accuracy too low ({acc:.2f})"

    # Save accuracy to a file for CML
    with open("result.log", "w") as f:
        f.write(f"Sanity Test Accuracy: {acc:.4f}\n")
