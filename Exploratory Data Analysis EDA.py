import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv(r"C:\Users\fawad\OneDrive\Desktop\advertising.csv")

# Print all available column names in your CSV to check spelling/case
print("Columns in your dataset:", df.columns.tolist())

print(df.describe())

# Replace 'YourColumnName' with an actual column from the printed list above
# print(df["YourColumnName"].value_counts(normalize=True))

# Correlation Matrix visualization setup
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
