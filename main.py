import math
import random
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# --- 1. Customer Dataset.py ---
random.seed(42)


def poisson_sample(lam):
  threshold = math.exp(-lam)
  count = 0
  product = 1.0
  while product > threshold:
    count += 1
    product *= random.random()
  return count - 1


n_samples = 1500
data = {
    "customer_id": [f"CUST_{i:04d}" for i in range(1, n_samples + 1)],
    "tenure_months": [random.randrange(1, 72) for _ in range(n_samples)],
    "monthly_charges": [
        round(random.uniform(20.0, 120.0), 2) for _ in range(n_samples)
    ],
    "total_charges": math.nan,
    "support_calls": [poisson_sample(1.5) for _ in range(n_samples)],
    "contract_type": random.choices(
        ["Month-to-Month", "One Year", "Two Year"],
        weights=[0.5, 0.3, 0.2],
        k=n_samples,
    ),
    "internet_service": random.choices(
        ["DSL", "Fiber Optic", "None"], weights=[0.4, 0.5, 0.1], k=n_samples
    ),
    "churn": random.choices([0, 1], weights=[0.75, 0.25], k=n_samples),
}

df = pd.DataFrame(data)
df.loc[df.sample(frac=0.05, random_state=42).index, "total_charges"] = math.nan
mask = df["total_charges"].isna()
df.loc[~mask, "total_charges"] = (
    df.loc[~mask, "tenure_months"] * df.loc[~mask, "monthly_charges"]
)
df.to_csv("customer_churn_raw.csv", index=False)
print(f"Dataset generated successfully with shape: {df.shape}")

# --- 2. Preprocess Data using Pandas.py ---
df = pd.read_csv("customer_churn_raw.csv")
df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())
df = df.drop(columns=["customer_id"])
df_encoded = pd.get_dummies(
    df, columns=["contract_type", "internet_service"], drop_first=True
)
X = df_encoded.drop(columns=["churn"])
y = df_encoded["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# --- 3. Machine-Learning Model.py ---
model = RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
print("Model training completed successfully.")

# --- 4. Trained Model Using Joblib.py ---
artifact = {"model": model, "feature_columns": list(X_train.columns)}
model_filename = "customer_churn_model.pkl"
joblib.dump(artifact, model_filename)
print(f"Model successfully exported to {model_filename}")

# --- 5. Model Accuracy.py ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")


# --- 6. Prediction Logic.py ---
def predict_churn(new_data: dict, artifact_path="customer_churn_model.pkl"):
  loaded_artifact = joblib.load(artifact_path)
  inf_model = loaded_artifact["model"]
  train_columns = loaded_artifact["feature_columns"]

  input_df = pd.DataFrame([new_data])
  if "total_charges" not in input_df or pd.isna(input_df["total_charges"].iloc[0]):
    input_df["total_charges"] = (
        input_df["tenure_months"] * input_df["monthly_charges"]
    )

  input_encoded = pd.get_dummies(
      input_df, columns=["contract_type", "internet_service"], drop_first=True
  )
  input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

  prediction = inf_model.predict(input_encoded)[0]
  probability = inf_model.predict_proba(input_encoded)[0][1]

  return {
      "churn_prediction": int(prediction),
      "churn_probability": float(round(probability, 4)),
      "status": "High Risk" if probability > 0.5 else "Low Risk",
  }


sample_customer = {
    "tenure_months": 5,
    "monthly_charges": 89.50,
    "total_charges": 447.50,
    "support_calls": 3,
    "contract_type": "Month-to-Month",
    "internet_service": "Fiber Optic",
}

prediction_result = predict_churn(sample_customer)
print("\nInference Output for Sample Customer:")
print(prediction_result)