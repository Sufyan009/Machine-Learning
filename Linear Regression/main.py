import matplotlib.pyplot as plt
import numpy as np
# import sklearn
from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# print(diabetes)
# print(diabetes.DESCR)
diabetes_X = diabetes.data
# print(diabetes_X)
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[:-30]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[:-30]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predict = model.predict(diabetes_X_test)
print("Mean Square error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predict))
print("Wight: ", model.coef_)
print("intercept: ", model.intercept_)

# plt.scatter(diabetes_X_train, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predict)
# plt.show()

# Mean Square error is:  3954.611332145007
# Wight:  [941.43097333]
# intercept:  153.39713623331644
