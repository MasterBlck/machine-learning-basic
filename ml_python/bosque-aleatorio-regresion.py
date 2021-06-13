#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:24:26 2020

@author: linuxlite
"""
#se importan las librerías a utilizar
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
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

############## PREPARAR LA DATA DE BOSQUES ALEATORIOS DE REGRESION#######################
#Seleccionamos solamente la columna 6 del dataset
X_bar = boston.data[:, np.newaxis, 5]

#definir los datos correspondientes a las etiquetas
y_bar = boston.target

#verificamos datos correspondientes
plt.scatter(X_bar, y_bar)
plt.show()

############# IMPLEMENTACIÓN DE BOSQUES ALEATORIOS DE REGRESION#########################

#Separa los datos de "train" en entrenamiento, y prueba(test) para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_bar, y_bar, test_size = 0.2)

#definir el algoritmo a utilizar
#n_estimators  --------> número estimado de árboles
bar = RandomForestRegressor(n_estimators=300, max_depth = 8)

#Entrena el modelo
#entrena el modelo
bar.fit(X_train, y_train)

#Realizo una predicción
Y_pred = bar.predict(X_test)

#Graficamos los datos de prueba junto con la prediccion
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, bar.predict(X_grid), color='red', linewidth=3)
plt.show()

print('DATDS DEL MODELO REGRESIÓN POLINOMIAL')
print()
print('Precisión del modelo:')
print(bar.score(X_train, y_train))
