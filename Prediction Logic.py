    def predict_churn(new_data: dict, artifact_path="customer_churn_model.pkl"):
        """Inference function for predicting customer churn from raw input dictionary."""
        # Load serialized artifact
        artifact = joblib.load(artifact_path)
        model = artifact["model"]
        train_columns = artifact["feature_columns"]

        # Convert input dict to DataFrame
        input_df = pd.DataFrame([new_data])

        # Feature engineering / imputation mirroring training pipeline
        if "total_charges" not in input_df or pd.isna(input_df["total_charges"].iloc[0]):
            input_df["total_charges"] = (
                input_df["tenure_months"] * input_df["monthly_charges"]
            )

        # One-hot encode categorical variables
        input_encoded = pd.get_dummies(
            input_df, columns=["contract_type", "internet_service"], drop_first=True
        )

        # Reindex to match exact training columns (fills missing dummy columns with 0)
        input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        result = {
            "churn_prediction": int(prediction),
            "churn_probability": float(round(probability, 4)),
            "status": (
                "High Risk" if probability > 0.5 else "Low Risk"
            ),
        }
        return result


    # --- Example Test Inference ---
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