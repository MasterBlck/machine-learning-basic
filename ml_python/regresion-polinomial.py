#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:19:29 2020

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


############## PREPARAR LA DATA DE REGRESION POLINOMIAL#######################
#Seleccionamos solamente la columna 6 del dataset
X_poli = boston.data[:, np.newaxis, 5]

#definir los datos correspondientes a las etiquetas
y_poli = boston.target

#verificamos datos correspondientes
plt.scatter(X_poli, y_poli)
plt.show()

############# IMPLEMENTACIÓN DE REGRESIÓN POLINOMIAL #########################
#Separa los datos de "train" en entrenamiento, y prueba(test) para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_poli, y_poli, test_size = 0.2)

#definiendo el grado del polinomio
from sklearn.preprocessing import PolynomialFeatures

#Se define el grado del polinomio
poli_reg = PolynomialFeatures(degree = 2)

#Se transforma las características existentes en características de mayor grado
X_train_poli = poli_reg.fit_transform(X_train)
X_test_poli = poli_reg.fit_transform(X_test)

#Se define el algoritmo a utilizar
pr = linear_model.LinearRegression()

#Entrenar el modelo
pr.fit(X_train_poli, y_train)

#Realizo una predicción
Y_pred_pr = pr.predict(X_test_poli)

#graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred_pr, color='red', linewidth=3)
plt.show()

print('DATDS DEL MODELO REGRESIÓN POLINOMIAL')
print()
print('Valor de la pendiente o coeficiente "a"')
print(pr.coef_)
print('Valor de la intersección o coeficiente "b"')
print(pr.intercept_)

print('La ecuación del modelo es igual a:')
print('y = ',pr.coef_, 'x ', pr.intercept_ )

print('Precisión del modelo:')
print(pr.score(X_train_poli, y_train))