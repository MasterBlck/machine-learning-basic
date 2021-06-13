#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:59:26 2020

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

#****************************Empezamos con la regresión lineal múltiple**********************
#seleccionamos sólo la columna 5, 6, y 7 de nuestro dataset
X_multiple = boston.data[:, 5:8]
print(X_multiple)


#Definir los datos correspondientes a las etiquetas
y_multiple = boston.target

#************************Separar lo s datos en entrenamiento y prueba*************************
#Separa los datos de "train" en entrenamiento, y prueba(test) para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size = 0.2)

#Definimos el algoritmo a utilizar, en este caso de regresión linear
lr_multiple = linear_model.LinearRegression()

#entrenamos el modelo
lr_multiple.fit(X_train, y_train)

#Realiza la predicción
Y_pred_multiple = lr_multiple.predict(X_test)

print('DATDS DEL MODELO REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a"')
print(lr_multiple.coef_)
print('Valor de la intersección o coeficiente "b"')
print(lr_multiple.intercept_)

print('La ecuación del modelo es igual a:')
print('y = ',lr_multiple.coef_, 'x ', lr_multiple.intercept_ )

print('Precisión del modelo:')
print(lr_multiple.score(X_train, y_train))

#no se grafica la regresión lineal múltiple