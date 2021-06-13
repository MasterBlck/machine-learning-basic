# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
import numpy as np

a = np.array([1, 2, 3])
print("Vector unidimensional")
print(a)
print()
b = np.array([(1, 2, 3), (4, 5, 6)])
print("Vector bidimensional")
print(b)
print()
print(b[0,1])

#array vs numpy
import sys
s = range(1000)
print("Resultado de lista de Python")
print(sys.getsizeof(5)*len(s))
print()


d = np.arange(1000)
print('Resultado de Numpy Array')
print(d.size * d.itemsize)

#---------------------------------Rendimiendo de ejecución de numpy

import time

SIZE = 1000000

L1 = range(SIZE)
L2 = range(SIZE)

A1 = np.arange(SIZE)
A2 = np.arange(SIZE)

start = time.time()
#Hace una suma de las listas uno a uno
result = [(x, y) for x, y in zip(L1, L2)]

print("Resultado de lista de python")
print((time.time()-start)*1000)
print()

start = time.time()
result = A1 + A2
print("Resultado de Numpy Array")
print((time.time()-start)*1000)



#------------test zip--------------
