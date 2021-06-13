#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:00:51 2020

@author: linuxlite
"""

import matplotlib.pyplot as plt

#Graficación básica
a = [3, 4, 5, 6]  #las x
b = [5, 6, 3, 4]  #las y 

plt.plot(a,b)
plt.show()

#-----------------------gráfica de 2 líneas, diferentes colores----------------
#línea 1
x1 = [3, 4, 5 ,6]
y1 = [5, 6, 3, 4]

#línea2
x2 = [2, 5, 8]
y2 = [3, 4, 3]

#para que se grafiquen las 2 líneas juntas se usa el mismo objeto plt
plt.plot(x1, y1, label='Línea 1', linewidth=4, color='blue')
plt.plot(x2, y2, label='Línea 2', linewidth=4, color='green')

#Definir título y nombres de los ejes
plt.title('Diagrama de líneas')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#Mostrar leyenda, cuadrícula y figura
plt.legend()
plt.grid()
plt.show()


#-----------------------gráfica de 2 barras, diferentes colores----------------
#definir datos
x1 = [0.25, 1.25, 2.25, 3.25, 4.25]
y1 = [10, 55, 80, 32, 40]

x2 = [0.75, 1.75, 2.75, 3.75, 4.75]
y2 = [42, 26, 10, 29, 66]

#Configurar características del gráfico de BARRAS
plt.bar(x1, y1, label='Datos 1', width=0.5, color='lightblue')
plt.bar(x2, y2, label='Datos 2', width=0.5, color='orange')

#Definir título y nombres de ejes
plt.title('Gráfico de barras')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#mostrar leyenda y figura
plt.legend()
plt.show()


#-----------------------gráfica de histogramas---------------------------------
#definir los datos
a = [22,55,62,45,21,22,34,42,42,4,2,102,95,85,55,110,120,70,65,55,111,115,80,75,65,54,44,43,42,48,19]
bins = [0,10,20,30,40,50,60,70,80,90,100]  #parecido a tener eje x

#configurar características del gráfico
plt.hist(a, bins, histtype='bar', rwidth=0.8, color='#ff5733')

#Definir título y nombres de ejes
plt.title('Histogramas')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#mostrar figura
plt.show()

#-----------------------gráfica de dispersión----------------------------------
#definir los datos
x1 = [0.25, 1.25, 2.25, 3.25, 4.25]
y1 = [10, 55, 80, 32, 40]

x2 = [0.75, 1.75, 2.75, 3.75, 4.75]
y2 = [42, 26, 10, 29, 66]

#configurar características del gráfico
plt.scatter(x1, y1, label='Datos 1', color='red')
plt.scatter(x2, y2, label='Datos 2', color='purple')

#Definir título y nombres de ejes
plt.title('Gráfico de dispersión')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#mostrar leyenda y figura
plt.legend()
plt.show()


#-----------------------gráfico circular---------------------------------------
#Definir los datos
dormir =[7,8,6,11,7]
comer = [2,3,4,3,2]
trabajar =[7,8,7,2,2]
recreación = [8,5,7,8,13]
divisiones = [7,2,2,13]
actividades = ['Dormir','Comer','Trabajar','Recreación']
colores = ['red','purple','blue','orange']

plt.pie(
        divisiones, 
        labels=actividades, 
        colors=colores, 
        startangle=90, 
        shadow=True, 
        explode=(0.1, 0, 0,0),
        autopct='%1.1f%%'
)

#Definir título
plt.title('Gráfico circular')

#Mostrar figura
plt.show()
