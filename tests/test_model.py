from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "heart_disease_model.joblib"
DATA_PATH = BASE_DIR / "data" / "heart_disease_clean.csv"

def test_dataset_exists():
    assert DATA_PATH.exists()

def test_model_exists():
    assert MODEL_PATH.exists()

def test_model_prediction():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"]).iloc[[0]]

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    assert prediction in [0, 1]
    assert len(probabilities) == 2
    assert 0 <= max(probabilities) <= 1
