#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:26:40 2020

@author: linuxlite
"""

#Código original:
#https://colab.research.google.com/drive/1vm1EAZ7lLRooZHqHTfUlul5ZXpGmb_SZ#scrollTo=vkVIchcshHDj
#referencia:
#https://www.youtube.com/watch?v=W8AeOXa_FqU

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

#Crear el dataset ---------------------------------------------------------------
make_circles? #ayuda de ..

n = 500 #Número de registros que tenemos en nuetsros datos
p = 2  #cuántas características tenemos sobre cada uno de los registros de nuestros datos

X, Y = make_circles(
            n_samples = n,
            factor = 0.5,      #factor de distancia entre los 2 círculos
            noise = 0.05   #hace que se distribuyan los datos de una manera pseudoaleatoria para simular dispersión de la data
       )
Y = Y[:, np.newaxis]
print(Y)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

#condicionando la salida con respecto a la variable Y
plt.scatter(X[ Y[:, 0] == 0 , 0], X[Y[:, 0] == 0 , 1], color = 'skyblue')
plt.scatter(X[ Y[:, 0] == 1 , 0], X[Y[:, 0] == 1 , 1], color = 'salmon')
plt.axis('equal')
plt.show()


#CLASE DE LA CAPA DE LA RED
class neural_layer():
    #n_conn número de conexiones entrantes a la capa
    #n_neur número de neuronas que hay en ésta capa
    #act_fn función de activación que se está ejecutando dentro de las neuronas de ésta capa
    def __init__(self, n_conn, n_neur, act_fn):
        self.act_fn = act_fn
        #inicializar los parámetros de la capa
        self.b = np.random.rand(1, n_neur) * 2 - 1 #va de -1 a 1 #se crea un vector que se refiere al parámetro BIAS, (inicializado en forma aleatoria)
        
        #matriz cuyos valores serán el número de conexiones y número de neuronas
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1 
        
#FUNCIONES DE ACTIVACION
#sigmoide
sigm = lambda x : 1 / (1 + np.e**(-x)) # función f(x) = 1 / (1 + np.e**(-x))

#visualización de la función
_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm(_x))
plt.grid()

#definición de tupla de funciones
sigm = (
            lambda x : 1 / (1 + np.e**(-x)), # función f(x) = 1 / (1 + np.e**(-x))
            lambda x : x * (1-x)             # derivada de la función anterior f'(x)
        )

#visualización de la función
_x = np.linspace(-5, 5, 100)
plt.plot(_x, sigm[0](_x), color='purple')  #ploteando la función f(x)
plt.plot(_x, sigm[1](_x), color='green')  #ploteando la derivada f'(x)
plt.grid()


relu = lambda x : np.maximum(0, x)
plt.plot(_x, relu(_x), color='green')  #ploteando la función relu

#----------------------------Creando las capas---------------------------------
#creación de capas de forma individual
#layer0 = neural_layer(p, 4, sigm)   #capa de 4 neuronas
#layer1 = neural_layer(p*2, 8, sigm)   #capa de 8 neuronas
#...

def create_neural_network(topology, act_fn):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_fn))
    
    return nn

topology = [p, 4, 8, 16, 8, 4, 1]
create_neural_network(topology, sigm)


#-----------------------Funciones para entrenar a la red neuronal--------------
topology = [p, 4, 8, 1]
neural_net = create_neural_network(topology, sigm)
l2_cost = (
            lambda Yp, Yr: np.mean((Yp - Yr) ** 2),  #Función de coste f(Yp, Yr)
            lambda Yp, Yr: (Yp - Yr)  #derivada de la Función de coste f'(Yp, Yr)
        )

def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    #Forward pass
    #forma individual
    #"""
    #z = X @ neural_net[0].W + neural_net[0].b # el @ multiplica las matrices
    #a = neural_net[0].act_fn(2)
    #"""
    out = [(None, X)]
    #forma iterativa
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b # el @ multiplica las matrices
        a = neural_net[l].act_fn[0](z)
        out.append((z, a))
        
    #print(out[-1][1])
    if train:
        # Backward pass
        deltas = []
        for l in reversed(range(0, len(neural_net))):
            z = out[l+1][0]
            a = out[l+1][1]
            
            if l == len(neural_net) - 1:
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_fn[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_fn[1](a))
                
            _W = neural_net[l].W
            
            # Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr   
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr
    
    return out[-1][1]

train(neural_net, X, Y, l2_cost, lr=0.5)
print()

#***************************************Visualización y test-*************************************
import time
from IPython.display import clear_output

neural_n = create_neural_network(topology, sigm)

loss = []

for i in range(2500):
    
    # Entrenemos a la red!
    pY = train(neural_n, X, Y, l2_cost, lr=0.05)
    if i % 25 == 0:
        print(pY)
        loss.append(l2_cost[0](pY, Y))
        res = 50

        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
    
        _Y = np.zeros((res, res))
        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]
        
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")
    
        plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c="salmon")
    
        clear_output(wait=True)
        plt.show()
        plt.plot(range(len(loss)), loss)
        plt.show()
        time.sleep(0.5)  