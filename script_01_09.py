# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:21:43 2021

@author: Ema
"""

##Imports scipy esta de mas
import numpy as np
from scipy import signal
from scipy.fft import fft
import matplotlib.pyplot as plt


##Cantidad de muestras
N=2000
fs=2000
##vector tiempo de 0 a 1, no incluido, con N saltos
t = np.linspace(0, 1, N, endpoint=False)

#senoidal de amplitud 1 y 1 hz de frecuencia
amp=1 #amplitud en volts
freq=1 #frecuencia en Hz
fase=0 #radianes
seno = amp*np.sin(2 * np.pi * freq * t + fase)

y = fft(seno)/N

# plt.figure(21)

# plt.stem(np.arange(len(t))/fs, np.abs(y)/N)

# plt.title('Modulo')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('Magnitud')
# #     notita = mpatches.Patch(label='xx')
# #     plt.legend(handles=[notita])
# plt.grid()

            
# plt.figure(22)

# plt.stem(np.arange(len(t))/fs, np.angle(y))
# #plt.show()

##2**(b-1) siendo b=bits

energia_tiempo=sum(seno**2)/N
energia_freq= sum(np.abs(y)**2)

varianza = np.var(seno)