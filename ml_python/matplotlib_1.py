#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:08:15 2020

@author: linuxlite
"""
#Báses de matplotlib
import matplotlib.pyplot as plt

#data examples
a = [1, 2, 3, 4]
b = [11, 22, 33, 44]

#graficando los datos
plt.plot(a,b, color='blue', linewidth=3, label='línea')
plt.legend()
plt.show()