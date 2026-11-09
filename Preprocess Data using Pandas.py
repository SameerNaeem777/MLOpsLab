from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("customer_churn_raw.csv")

# 1. Handle Missing Values (Impute missing total_charges with median)
df["total_charges"] = df["total_charges"].fillna(df["total_charges"].median())

# 2. Drop unique identifier columns not used for modeling
df = df.drop(columns=["customer_id"])

# 3. One-Hot Encode Categorical Variables
df_encoded = pd.get_dummies(
    df, columns=["contract_type", "internet_service"], drop_first=True
)

# 4. Define Features (X) and Target (y)
X = df_encoded.drop(columns=["churn"])
y = df_encoded["churn"]

# 5. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")