import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import os

matplotlib.use('Agg')

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Housing.csv")
export_dir = os.path.join(script_dir, "export")
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Load data
df = pd.read_csv(file_path)

# Use multiple features to see the effect of regularization better than just 1 feature
features = ['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']
X = df[features]
Y = df['price']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---- Ridge Regression (L2) ----
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, Y_train)
ridge_pred = ridge_reg.predict(X_test)
ridge_mse = mean_squared_error(Y_test, ridge_pred)
ridge_r2 = r2_score(Y_test, ridge_pred)

print("Ridge Regression Results:")
print(f"Mean Squared Error: {ridge_mse:.2f}")
print(f"R2 Score: {ridge_r2:.2f}")

# ---- Lasso Regression (L1) ----
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, Y_train)
lasso_pred = lasso_reg.predict(X_test)
lasso_mse = mean_squared_error(Y_test, lasso_pred)
lasso_r2 = r2_score(Y_test, lasso_pred)

print("\nLasso Regression Results:")
print(f"Mean Squared Error: {lasso_mse:.2f}")
print(f"R2 Score: {lasso_r2:.2f}")

# Plot Actual vs Predicted for both
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(Y_test, ridge_pred, color='blue', alpha=0.5)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.title(f'Ridge Regression (R2={ridge_r2:.2f})')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 2, 2)
plt.scatter(Y_test, lasso_pred, color='green', alpha=0.5)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
plt.title(f'Lasso Regression (R2={lasso_r2:.2f})')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Unique save function
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)

output_path = get_unique_filename(export_dir, "ridge_lasso_plot.png")
plt.savefig(output_path)
print(f"Plot saved as {output_path}")
plt.close()
