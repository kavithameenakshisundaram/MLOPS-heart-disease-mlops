from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

dataset = fetch_ucirepo(id=45)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()

df = pd.concat([X, y], axis=1)

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

target_column = df.columns[-1]
df["target"] = (df[target_column] > 0).astype(int)

if target_column != "target":
    df = df.drop(columns=[target_column])

path = DATA_DIR / "heart_disease_clean.csv"
df.to_csv(path, index=False)

print(f"Saved dataset to: {path}")
print(f"Shape: {df.shape}")
