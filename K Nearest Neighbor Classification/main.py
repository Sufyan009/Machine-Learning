# Loading Required Modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
"""Importing & Loading Datasets"""

# Printing Description and features
features = iris.data
labels = iris.target
# print(features[0], labels[0])
# print(iris.DESCR)

# training the classifiers
clf = KNeighborsClassifier()
clf.fit(features, labels)

predict = clf.predict([[0.1, 0.6, 1, 0.0009]])
print(predict)
