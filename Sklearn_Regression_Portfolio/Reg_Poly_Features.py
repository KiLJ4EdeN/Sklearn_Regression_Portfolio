import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Weather.csv')
X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
y = y.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True, order='C')
poly.fit_transform(X_train)
poly.transform(X_test)
regressor = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(regressor.intercept_)
print(regressor.coef_)

