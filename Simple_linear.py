import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Housing.csv")
df = pd.read_csv(file_path)
Y = df['price']
X = df['lotsize']
X = X.to_numpy().reshape(len(X), 1)
Y = Y.to_numpy().reshape(len(Y), 1)

# Split the data into training/testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=250, random_state=42)
plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

plt.plot(X_test, regr.predict(X_test), linewidth=3, color='red')
export_dir = os.path.join(script_dir, "export")
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Unique save function
def get_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return os.path.join(directory, new_filename)

output_path = get_unique_filename(export_dir, "regression_plot.png")
plt.savefig(output_path)
print(f"Plot saved as {output_path}")