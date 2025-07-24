import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
csv_path = 'premier_league_attackers_2023-24.csv'
# Read CSV and treat common missing value strings as NaN
# skip initial blank lines automatically
raw_df = pd.read_csv(csv_path, na_values=['N/A', 'NA', ''])

# Drop columns that are entirely NaN
df = raw_df.dropna(axis=1, how='all').copy()

# Separate features and automatically determine the target column
numeric_df = df.select_dtypes(include='number')
if len(numeric_df.columns) < 2:
    # Fallback to using the last column as the target when not enough numeric data
    target_col = df.columns[-1]
else:
    corr_matrix = numeric_df.corr().abs()
    avg_corr = corr_matrix.sum() - 1  # exclude self-correlation
    avg_corr = avg_corr / (len(numeric_df.columns) - 1)
    target_col = avg_corr.idxmax()

features = [c for c in df.columns if c != target_col]
X_raw = df[features]
y = df[target_col]

# Preprocess features: fill missing values and encode categorical columns
numeric_features = X_raw.select_dtypes(include='number').columns.tolist()
categorical_features = X_raw.select_dtypes(exclude='number').columns.tolist()

# Simple imputation: fill numeric with median, categorical with mode
for col in numeric_features:
    median = X_raw[col].median()
    X_raw[col] = X_raw[col].fillna(median)
for col in categorical_features:
    mode = X_raw[col].mode().iat[0]
    X_raw[col] = X_raw[col].fillna(mode)

# Column transformer for encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X_raw)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_processed, y)

# Predictions
predictions = model.predict(X_processed)

# Create performance index scaled between 0 and 100
index_scaler = MinMaxScaler(feature_range=(0, 100))
performance_index = index_scaler.fit_transform(predictions.reshape(-1, 1)).flatten()

# Add new column to DataFrame
output_df = df.copy()
output_df['Performance_Index_0_100'] = performance_index

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y)), y, label='Actual')
plt.scatter(range(len(y)), predictions, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel(target_col)
plt.title('Actual vs Predicted')
plt.legend()
plt.tight_layout()
plot_path = 'actual_vs_predicted.png'
plt.savefig(plot_path)

# Print regression equation
coefs = model.coef_
intercept = model.intercept_

# Get feature names after preprocessing
encoded_cat = preprocessor.named_transformers_['cat']
encoded_cat_names = encoded_cat.get_feature_names_out(categorical_features) if categorical_features else []
feature_names = numeric_features + list(encoded_cat_names)

formula_terms = [f"({coef:.4f} * {name})" for coef, name in zip(coefs, feature_names)]
formula = " + ".join(formula_terms)
print(f"Linear Regression Formula:\n{target_col} = {intercept:.4f} + " + formula)

# Export DataFrame with new column
output_csv = 'premier_league_attackers_2023-24_with_performance_index.csv'
output_df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv} and plot saved to {plot_path}")
