import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Weather.csv')

#print(dataset.shape)
#rint(dataset.describe())

#dataset.plot(x='MinTemp', y='MaxTemp', style='o')
#plt.title('MinTemp vs MaxTemp')
#plt.xlabel('MinTemp')
#plt.ylabel('MaxTemp')
#plt.show()
#plt.figure(figsize=(15, 10))
#plt.tight_layout()
#seabornInstance.distplot(dataset['MaxTemp'])
#plt.show()

X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
#0.92 percent changes on every step of min_temp to max_temp
print(regressor.coef_)

y_pred = regressor.predict(X_test)
print(y_pred.shape)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
