# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
tree.score(x_test, y_test)
tree.score(x_train, y_train)

export_graphviz(tree, out_file='arbol.dot',class_names=iris.target_names, feature_names=iris.feature_names, impurity=False, filled=True)

with open('arbol.dot') as mFile:
    dot_graph = mFile.read()

#Despliega el arbol de desiciones
graphviz.Source(dot_graph)

#grafica la característica que más importancia tiene
caract = iris.data.shape[1]
plt.barh(range(caract), tree.feature_importances_)
plt.yticks(np.arange(caract), iris.feature_names)
plt.xlabel('Importancia de las características')
plt.ylabel('Característica')
plt.show()

tree = DecisionTreeClassifier(max_depth=3) #maxima profundidad del árbol
tree.fit(x_train, y_train)
tree.score(x_test, y_test)
tree.score(x_train, y_train)

#desplegando datos con matplotlib
n_classes = 3
plot_colors = 'bry'
plot_step = 0.02
for pairidx, pair in enumerate([
    [0,1], [0,2], [0,3],
    [1,2], [1,3], [2,3]]):
    x = iris.data[:, pair]
    y = iris.target

    #Entrenar al algoritmo
    clf = DecisionTreeClassifier(max_depth=3).fit(x,y)

    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = x[:, 0].min() - 1, x[:,0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step)
                         np.arange(y_min, y_max, plot_step))
    
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = z.reshape(xx.shape) #warning!!!
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    #plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(x[idx, 0], x[idx, 1], c=color, label=iris.target_names[i]
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.suptitle("Ejemplo de clasificador de árboles")
plt.legend()
plt.show()