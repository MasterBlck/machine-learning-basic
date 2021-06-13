#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:04:56 2020

@author: linuxlite
"""
#se importan las librerías a utilizar
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
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

############## PREPARAR LA DATA DE Arboles de decision DE REGRESION#######################
#Seleccionamos solamente la columna 6 del dataset
X_adr = boston.data[:, np.newaxis, 5]

#definir los datos correspondientes a las etiquetas
y_adr = boston.target

#verificamos datos correspondientes
plt.scatter(X_adr, y_adr)
plt.show()

############# IMPLEMENTACIÓN DE ARBOLES DE DECISIÓN DE REGRESION#########################

#Separa los datos de "train" en entrenamiento, y prueba(test) para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_adr, y_adr, test_size = 0.2)

#definir el algoritmo a utilizar
adr = DecisionTreeRegressor(max_depth = 5)

#entrena el modelo
adr.fit(X_train, y_train)

#Realizo una predicción
Y_pred = adr.predict(X_test)

#Graficamos los datos de prueba junto con la prediccion
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, adr.predict(X_grid), color='red', linewidth=3)
plt.show()

print('DATDS DEL MODELO REGRESIÓN POLINOMIAL')
print()
print('Precisión del modelo:')
print(adr.score(X_train, y_train))
