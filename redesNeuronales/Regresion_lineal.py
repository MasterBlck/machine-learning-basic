#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:16:45 2020

@author: linuxlite
"""
#Video de referencia:
#https://www.youtube.com/watch?v=w2RJ1D6kz-o
#Algoritmo de mínimos cuadrados ordinarios

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

#Cargamos la librería
boston = load_boston()
#print(boston.DESCR)

#obteniendo la matriz, numero promedio de habitaciones por cada barrio de la ciudad de Boston
X = np.array(boston.data[:, 5]) #Toda la fila de la columna 5  X ----->número de habitaciones
Y = np.array(boston.target)                                   #Y  -----> precio aproximado

#graficando los puntos
plt.scatter(X,Y, alpha=0.3)
plt.xlabel('Número de habitaciones')
plt.ylabel('Precio medio')
#plt.show()

#Añadimos columnas de 1s para término independiente, al mismo tiempo que calculamos la traspuesta
X = np.array([np.ones(506), X]).T

#fórmula de minimizar el error cuadrático medio (MCO)
#beta = (X^T * X)^-1  *  X^T*Y  
B = np.linalg.inv(X.T @ X) @ X.T @ Y  #multiplicación matricial con @ tiene que ser invertido con np.linalg.inv

#dibujando la línea
plt.plot([4, 9], [B[0] + B[1] * 4, B[0] + B[1] * 9], c="red")
plt.show()