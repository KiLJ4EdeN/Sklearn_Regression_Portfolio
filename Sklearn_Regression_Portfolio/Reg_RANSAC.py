import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.linear_model import RANSACRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Weather.csv')
X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
y = y.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = RANSACRegressor(base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None,
                            is_model_valid=None, max_trials=100, max_skips=inf, stop_n_inliers=inf,
                            stop_score=inf, stop_probability=0.99, loss='absolute_loss', random_state=None)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(regressor.intercept_)
print(regressor.coef_)
