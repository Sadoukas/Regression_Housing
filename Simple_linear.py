import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
df = pd.read_csv("Housing.csv")
Y = df['price']
X = df['lotsize']
X = X.to_numpy().reshape(len(X), 1)
Y = Y.to_numpy().reshape(len(Y), 1)

X_train = X[:-250]
X_test = X[-250:]
Y_train = Y[:-250]
Y_test = Y[-250:]
plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

plt.plot(X_test, regr.predict(X_test), linewidth=3, color='red')
plt.savefig("regression_plot.png")
print("Plot saved as regression_plot.png")