import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "heart_disease_clean.csv"
MODEL_DIR = BASE_DIR / "artifacts" / "models"
FIGURE_DIR = BASE_DIR / "artifacts" / "figures"
MLRUN_DIR = BASE_DIR / "mlruns"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MLRUN_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["target"])
y = df["target"]

categorical_features = [
    c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    if c in X.columns
]
numerical_features = [
    c for c in ["age", "trestbps", "chol", "thalach", "oldpeak"]
    if c in X.columns
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

search.fit(X_train, y_train)

model = search.best_estimator_
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, pred),
    "precision": precision_score(y_test, pred, zero_division=0),
    "recall": recall_score(y_test, pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, prob)
}

model_path = MODEL_DIR / "heart_disease_model.joblib"
joblib.dump(model, model_path)

(BASE_DIR / "artifacts" / "model_metadata.json").write_text(
    json.dumps(
        {
            "model": "Random Forest",
            "best_parameters": search.best_params_,
            "metrics": metrics,
            "features": X.columns.tolist()
        },
        indent=2
    )
)

mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
mlflow.set_experiment("heart-disease-classification")

with mlflow.start_run(run_name="random-forest-final"):
    mlflow.log_params(search.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
    )

print("Best parameters:", search.best_params_)
print("Metrics:", metrics)
print("Model:", model_path)
