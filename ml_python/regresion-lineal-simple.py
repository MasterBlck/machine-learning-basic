#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:10:03 2020

@author: linuxlite
"""

#se importan las librerías a utilizar
import numpy as np
from sklearn import datasets, linear_model
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

#seleccionamos sólo la columna 5 de nuestro dataset
X = boston.data[:, np.newaxis, 5]

#Definir los datos correspondientes a las etiquetas
y = boston.target

#graficamos los datos correspondientes con una gráfica de dispersión
plt.scatter(X, y)
plt.xlabel('Número de habitaciones')
plt.ylabel('Precio medio')
plt.show()

#Separa los datos de "train" en entrenamiento, y prueba(test) para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Definimos el algoritmo a utilizar, en este caso de regresión linear
lr = linear_model.LinearRegression()

#entrenamos el modelo
lr.fit(X_train, y_train)

#Realiza la predicción
Y_pred = lr.predict(X_test)

#graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresión lineal simple')
plt.xlabel('Número de habitaciones')
plt.ylabel('Precio medio')
plt.show()

print('DATDS DEL MODELO REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a"')
print(lr.coef_)
print('Valor de la intersección o coeficiente "b"')
print(lr.intercept_)

print('La ecuación del modelo es igual a:')
print('y = ',lr.coef_, 'x ', lr.intercept_ )

print('Precisión del modelo:')
print(lr.score(X_train, y_train))