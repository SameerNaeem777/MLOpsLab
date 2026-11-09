from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn_raw.csv"
MODEL_PATH = ROOT / "models" / "customer_churn_model.pkl"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"{DATA_PATH} is missing. Run scripts/generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    if df["total_charges"].notna().sum() == 0:
        raise ValueError("total_charges is entirely missing; regenerate the dataset.")
    df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
    df = df.drop(columns=["customer_id"])
    encoded = pd.get_dummies(
        df, columns=["contract_type", "internet_service"], drop_first=True, dtype=int
    )
    features = encoded.drop(columns=["churn"])
    target = encoded["churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
    )
    model.fit(x_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": list(x_train.columns)}, MODEL_PATH
    )
    print(f"Trained on {len(x_train)} rows; held out {len(x_test)} rows.")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
