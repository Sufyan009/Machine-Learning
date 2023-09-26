from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)

clf = LinearRegression()
clf.fit(X, y)
eg = clf.predict([[2.6]])
# print(eg)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# print(X_new)
y_prob = clf.predict_proba(X_new)


plt.plot(X_new, y_prob)
plt.show()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])
