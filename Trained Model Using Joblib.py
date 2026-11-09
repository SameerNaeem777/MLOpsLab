import joblib

# Bundle the model and the required feature columns together
artifact = {
    "model": model,
    "feature_columns": list(X_train.columns),
}

# Export to disk
model_filename = "customer_churn_model.pkl"
joblib.dump(artifact, model_filename)
print(f"Model successfully exported to {model_filename}")