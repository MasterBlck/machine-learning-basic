#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:17:44 2020

@author: linuxlite
"""
import pandas as pd
import numpy as np

#construcción básica de un dataframe
data = np.array([['', 'Col1', 'Col2'], ['Fila1', 11,22], ['Fila2', 33, 44]])
df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
print(df)

#Data frame básico con enteros
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
df = pd.DataFrame(arr)
print(df)
print("Forma del dataframe")
print(df.shape)
print("Altura del dataframe")
print(len(df.index))

#Sacar estadísticas
stadistics = df.describe()
print(stadistics)

#conocer la media de todas las columnas
print("Media de las columnas dataframe")
print(df.mean())

#Correlacion de dataframe
print("Correlacion de dataframe")
print(df.corr())

print("Conteo de datos de cada columna del dataframe")
print(df.count())

print("Valos más alto de cada columna del dataframe")
print(df.max())

print("Valos más bajo de cada columna del dataframe")
print(df.min())

print("Mediana de cada columna del dataframe")
print(df.median())

print("Desviación estándar de cada columna del dataframe")
print(df.std())

print("Seleccionando primera columna del dataframe")
print(df[0])

print("Seleccionando dos columnas del dataframe")
print(df[[0, 1]])

print("Seleccionando valor de la primera fila y última columna del dataframe")
print(df.iloc[0][2])

print("Seleccionando primera FILA del dataframe")
print(df.loc[0])


print("Seleccionando primera FILA del dataframe por iloc")
print(df.iloc[0, :])

#Creación de series
series = pd.Series({
            "Argentina" : "Buenos Aires",
            "Chile"     : "Santiago",
            "Colombia"  : "Bogotá",
            "Perú"      : "Lima"
        })
print("Series")
print(series)
print(series["Argentina"])
print(series[1])



#-------------------------------Importando archivos a pandas
df = pd.read_csv("train.csv")

#verificar si hay datos nulos en dataframe
print("Datos nulos en dataframe")
print(df.isnull())

print("Suma datos nulos en dataframe")
print(df.isnull().sum())

