#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:46:29 2020

@author: linuxlite
"""
import numpy as np

#CREACION de matrices
#creación de matrices con unos
unos = np.ones((3,4))
print(unos)

#creación de matrices con zeros
zeros = np.zeros((3,4))
print(zeros)

#creación de matrices con números aleatorios
m_aleatorio = np.random.random((5,5))
print(m_aleatorio)

#creación de matriz vacía
vacia = np.empty((4,4))
print(vacia)

#creación de matriz con un solo valor en todos las pocisiones
full = np.full((2,2), 8)
print(full)

#creación de matriz con espacios uniformes
espacio1 = np.arange(0, 30, 5)
print(espacio1)
espacio2 = np.arange(0, 2, 0.5)
print(espacio2)

#creación de matriz identidad
identidad1 = np.eye(4,4)
print(identidad1)
identidad2 = np.identity(4)
print(identidad2)

#-----------------------------Inspecciónde matrices------------------------
print(identidad1.ndim)   #imprime 2 por que se usa arreglo bidimensional
print(full.dtype)

#conocer el tamaño y forma de la matriz
a = np.array([1,2,3,4,5,6])
print(a.size)
print(a.shape)


#cambio de forma de una matriz
a = np.array([[8, 9, 10], [11, 12, 13]])
print(a)
a = a.reshape(3,2)
print(a)

#Extraer los valores de todas las filas ubicadas en la columna 3
a = np.array([[1,2,3,4],[3,4,5,6]])
print(a[0:,2])


#--------------------------Operaciones matemáticas básicas-------------------
a = np.array([2,4,8])
print(a.min())
print(a.max())
print(a.sum())


#calcular raiz cuadrada u desviación estándar
a = np.array([[1,2,3], [3,4,5]])
print(np.sqrt(a))
print(np.std(a))

#suma, resta, multiplicación y división
x = np.array([[1,2,3], [3,4,5]])
y = np.array([[1,2,3], [3,4,5]])
print(x + y)
print(x - y)
print(x * y)
print(x / y)