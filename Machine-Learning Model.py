import sklearn.ensemble

# Initialize and train the model
model = sklearn.ensemble.RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

print("Model training completed successfully.")