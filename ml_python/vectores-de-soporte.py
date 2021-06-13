#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:14:09 2020

@author: linuxlite
"""

#se importan las librerías a utilizar
import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


###################Preparar la data#############################3
boston = datasets.load_boston()
print(boston)
print()


print('Información en el dataset')
print(boston.keys())

print('Características del dataset')
print(boston.DESCR)

print('Cantidad de datos que hay en el dataset')
print(boston.data.shape)

#VErifica la información de las columnas
print('Nombres columnas')
print(boston.feature_names)

############## PREPARAR LA DATA DE VECTORES DE SOPORTE DE REGRESION#######################
#Seleccionamos solamente la columna 6 del dataset
X_svr = boston.data[:, np.newaxis, 5]

#definir los datos correspondientes a las etiquetas
y_svr = boston.target

#verificamos datos correspondientes
plt.scatter(X_svr, y_svr)
plt.show()

############# IMPLEMENTACIÓN DE VECTORES DE SOPORTE DE REGRESION#########################

#Separa los datos de "train" en entrenamiento, y prueba(test) para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_svr, y_svr, test_size = 0.2)

#definir el algoritmo a utilizar
svr = SVR(kernel='linear', C=1.0, epsilon=0.2)

#entrena el modelo
svr.fit(X_train, y_train)

#Realizo una predicción
Y_pred = svr.predict(X_test)

#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.show()


print('DATDS DEL MODELO REGRESIÓN POLINOMIAL')
print()
print('Precisión del modelo:')
print(svr.score(X_train, y_train))