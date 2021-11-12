# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:34:13 2021

@author: Ema
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft, fftshift
import matplotlib.patches as mpatches
from pandas import DataFrame
from IPython.display import HTML

#######################################################################################################################
#%% Inicio de la simulación
#######################################################################################################################
plt.close('all')

# Datos generales de la simulación
fs = 1000 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras

figura=0
 
ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

t=np.arange(0,1,ts)
f=np.arange(0,fs,df)
####################
#%%Armo la se;al
####################
m=200
N= np.array([10,50,100,250, 500, 1000, 5000])
areas=np.empty(shape=(1,200))
sigma_2=2


for i in range(len(N)):
    #plt.figure(figura)
    ##Armo mi espacio de N items y m iteraciones
    espacio_muestral=np.random.normal(loc=0,scale=np.sqrt(sigma_2),size=(N[i],m))
    t=np.arange(0,1,N[i])
    f=np.arange(0,fs,fs/N[i])
    #calculo el periodograma
    Pp=(1/N[i])*np.abs((fft(espacio_muestral,axis=0)))**2
    #hallo el valor esperado para cada frecuencia
    E_Pp=np.mean(Pp,axis=1)
    #Sumo todos los valores esperados
    area=np.mean(E_Pp)
    print('Sesgo=',area,'-2=',area-sigma_2)
    
    var_period=np.var(Pp,axis=1)
    varianza_prom=np.mean(var_period)
    print('Varianza promedio=',varianza_prom)
    #plt.plot(f,E_Pp)
    #plt.plot(f,areas)
    #figura+=1
#plt.plot(t,x[:,1])

#periodograma=np.abs(1/N * fft(x))**2