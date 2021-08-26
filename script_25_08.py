# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:18:56 2021

@author: Ema
"""
##Imports scipy esta de mas
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

##Cantidad de muestras
N=500
##vector tiempo de 0 a 1, no incluido, con 500 saltos
t = np.linspace(0, 1, N, endpoint=False)
#senoidal de amplitud 1 y 1 hz de frecuencia
sig1 = np.sin(1*2 * np.pi * t)
#visualizacion de la senoidal
#plt.plot(t, sig1)
#inicializacion de lo que va a ser mi vector de salida para la DFT
y = np.array([0+0j]).repeat(N)
#Otro intento de inicializacion
#y=np.array(sig1 * np.exp(-2j * np.pi * 1 * np.arange(n)/n),dtype=complex)

#aplico la formula que vimos en clase
# esto: https://aulasvirtuales.frba.utn.edu.ar/pluginfile.php/1310601/course/section/122726/2-Frequency%20analysis%20and%20FFT.pdf
#y un for para que lo haga N veces
for k in range(N):
    y[k] = np.sum(sig1 * np.exp(-2j * np.pi * k * np.arange(N)/N))
      
#Esto ultimo tengo dudas, lo hago porque Y tiene por indice [0;499]
#pero conceptualmente no se si es correcto
#y grafico el modulo porque sino pierdo la parte imaginaria que es donde esta mi informacion
plt.plot(np.arange(N), np.absolute(y))
