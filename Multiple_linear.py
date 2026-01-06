import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
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

# Select features and target
# Using multiple numeric features
features = ['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']
X = df[features]
Y = df['price']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train model
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

# Predict
Y_pred = regr.predict(X_test)

# Evaluate
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Multiple Linear Regression Results:")
print(f"Features: {features}")
print(f"Coefficients: {regr.coef_}")
print(f"Intercept: {regr.intercept_}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, color='blue', alpha=0.5)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2) # Identity line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Multiple Linear Regression: Actual vs Predicted')

# Unique save function
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)

output_path = get_unique_filename(export_dir, "multiple_linear_plot.png")
plt.savefig(output_path)
print(f"Plot saved as {output_path}")
plt.close()
