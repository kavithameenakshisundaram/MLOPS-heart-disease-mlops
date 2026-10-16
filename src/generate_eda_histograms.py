from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/heart_disease_clean.csv")
OUTPUT_PATH = Path("artifacts/figures/numerical_feature_histograms.png")
FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for index, feature in enumerate(FEATURES):
        values = pd.to_numeric(df[feature], errors="coerce").dropna()
        axes[index].hist(values, bins=15, edgecolor="black")
        axes[index].set_title(feature)
        axes[index].set_xlabel(feature)
        axes[index].set_ylabel("Frequency")
        axes[index].grid(axis="y", alpha=0.25)

    axes[5].axis("off")
    fig.suptitle(
        "Heart Disease Dataset: Numerical Feature Distributions",
        fontsize=15,
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Dataset shape: {df.shape}")
    print(f"Histogram image saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
