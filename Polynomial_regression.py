import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

# Use lotsize for 2D visualization
X = df[['lotsize']]
Y = df['price']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Polynomial Transformation (Degree 2 and 3 usually sufficient for demo)
degree = 2
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, Y_train)

# Predict
Y_pred = poly_reg.predict(X_test_poly)

# Sort for plotting smooth curve
X_range = np.linspace(X['lotsize'].min(), X['lotsize'].max(), 100).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
Y_range_pred = poly_reg.predict(X_range_poly)

# Evaluate
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Polynomial Regression (Degree {degree}) Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Test Data')
plt.plot(X_range, Y_range_pred, color='blue', linewidth=3, label=f'Polynomial Model (deg={degree})')
plt.xlabel('Lot Size')
plt.ylabel('Price')
plt.title(f'Polynomial Regression (Degree {degree})')
plt.legend()

# Unique save function
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)

output_path = get_unique_filename(export_dir, "polynomial_regression_plot.png")
plt.savefig(output_path)
print(f"Plot saved as {output_path}")
plt.close()
