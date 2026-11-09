
import math
import pandas as pd  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
import random

# Set random seed for reproducibility
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
    "monthly_charges": [round(random.uniform(20.0, 120.0), 2) for _ in range(n_samples)],
    "total_charges": math.nan,  # Will introduce some missing values for cleaning step
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

# Introduce missing values in total_charges intentionally to demonstrate cleaning
df.loc[df.sample(frac=0.05, random_state=42).index, "total_charges"] = math.nan
# Fill the rest logically: tenure * monthly charges with some noise
mask = df["total_charges"].isna()
df.loc[~mask, "total_charges"] = (
    df.loc[~mask, "tenure_months"] * df.loc[~mask, "monthly_charges"]
)

# Save raw data
df.to_csv("customer_churn_raw.csv", index=False)
print(f"Dataset generated successfully with shape: {df.shape}")


