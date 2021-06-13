import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Algoritmo de vecinos cercanos

#Getting the dataset
iris = load_iris()  # the datatype returned is 'sklearn.datasets.base.Bunch'

type(iris)

iris.keys()

iris['data']

iris['target_names']

iris['target']

iris['feature_names']

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'])

X_train.shape

y_train.shape


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

knn.score(X_test, y_test)

knn.predict([[1.2, 3.4,5.6,1.1]])

iris.target_names 