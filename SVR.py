import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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

# Use lotsize for visualization
X = df[['lotsize']]
Y = df['price'].values.reshape(-1, 1) # Reshape for scaler

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for SVR)
sc_X = StandardScaler()
sc_Y = StandardScaler()

X_train_scaled = sc_X.fit_transform(X_train)
Y_train_scaled = sc_Y.fit_transform(Y_train)

# Test data must be transformed using the same scalers
X_test_scaled = sc_X.transform(X_test)
Y_test_scaled = sc_Y.transform(Y_test)

# Train SVR Model
# Kernel options: 'linear', 'poly', 'rbf', 'sigmoid'
regressor = SVR(kernel='rbf')
regressor.fit(X_train_scaled, Y_train_scaled.ravel())

# Predict
Y_pred_scaled = regressor.predict(X_test_scaled)
Y_pred = sc_Y.inverse_transform(Y_pred_scaled.reshape(-1, 1))

# Evaluate
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Support Vector Regression (SVR) Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualization
X_grid = np.arange(X['lotsize'].min(), X['lotsize'].max(), 100).reshape(-1, 1)
X_grid_scaled = sc_X.transform(X_grid)
Y_grid_pred_scaled = regressor.predict(X_grid_scaled)
Y_grid_pred = sc_Y.inverse_transform(Y_grid_pred_scaled.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Test Data')
plt.plot(X_grid, Y_grid_pred, color='red', linewidth=2, label='SVR (RBF Kernel)')
plt.xlabel('Lot Size')
plt.ylabel('Price')
plt.title('Support Vector Regression (SVR)')
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

output_path = get_unique_filename(export_dir, "svr_plot.png")
plt.savefig(output_path)
print(f"Plot saved as {output_path}")
plt.close()
