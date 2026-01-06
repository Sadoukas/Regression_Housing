import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
Y = df['price']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model (using 100 trees)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, Y_train)

# Predict
Y_pred = regressor.predict(X_test)

# Evaluate
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Random Forest Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualization
X_grid = np.arange(X['lotsize'].min(), X['lotsize'].max(), 50).reshape(-1, 1)
Y_grid_pred = regressor.predict(X_grid)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='black', label='Test Data')
plt.plot(X_grid, Y_grid_pred, color='orange', linewidth=2, label='Random Forest')
plt.xlabel('Lot Size')
plt.ylabel('Price')
plt.title('Random Forest Regression')
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

output_path = get_unique_filename(export_dir, "random_forest_plot.png")
plt.savefig(output_path)
print(f"Plot saved as {output_path}")
plt.close()
