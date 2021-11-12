# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:54:01 2021

@author: Ema
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
import matplotlib.patches as mpatches
from scipy.fft import fft

N = 300 # muestras
fs= 500 #Hz
B = [4,8,16] # bits
V_f=2 # volts
## Declaro los pasos para cada bit distinto
q = [0]*len(B)
for i in range(len(B)):
    q[i] = V_f/2**(B[i]-1)


##vector tiempo de 0 a 1, no incluido, con N saltos
t = np.linspace(0, 1, N, endpoint=False)

##Noise
noise=[[0]*N]*len(B)
for i in range(len(B)):
    noise[i] = np.random.normal(0, q[i]/2, len(t))

#senoidal de amplitud 1 y 1 hz de frecuencia
amp=1 #amplitud en volts
freq=1 #fs/N #frecuencia en Hz
fase=0 #radianes
seno_cont=[[0]*N]*len(B)
for i in range(len(B)):
    seno_cont[i] = amp*np.sin(2 * np.pi * freq * t + fase) + noise[i]


e=[0]*len(B)
corr=[0]*len(B)
valor_medio=[0]*len(B)
muestreada=[[0]*N]*len(B)
DEP=[[0]*N]*len(B)

##Analizo el ruido
for i in range(len(B)):
    DEP[i]=np.abs(fft(noise[i]))**2
    #energia_freq= sum(np.abs(noise[i])**2)
##Genero y muestro mi señal muestreada contra la original
for i in range(len(B)):
    muestreada[i]=np.round(seno_cont[i]/q[i])*q[i]
    plt.figure(i)
    plt.plot(t,seno_cont[i])
    plt.stem(t, muestreada[i])
    plt.title('Continua vs Muestreada con Bits='+str(B[i]))
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Volts [V]')
    notita = mpatches.Patch(label='Resolucion:'+str(q[i]))
    plt.legend(handles=[notita])
##Genero y muestro mi error
for i in range(len(B)):
    e[i]=seno_cont[i]-muestreada[i]
    plt.figure('1-'+str(i))
    plt.stem(t,e[i])
    plt.title('Error'+str(i))
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Volts [V]')
##correlacion del error
for i in range(len(B)):
    corr[i] = sig.correlate(e[i], e[i], mode='same')
    plt.figure('2-'+str(i))
    plt.stem(t,corr[i])
    plt.title('Correlacion Error'+str(i)+'-Error'+str(i))
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Volts [V]')
##ploteo el histograma del error entre mi senial original y 
for i in range(len(B)):
    plt.figure('3-'+str(i))
    count, bins, ignored=plt.hist(e[i])
    plt.title('Histograma del Error')
    plt.xlabel('Valor de la variable')
    plt.ylabel('Conteo')
##ploteo la densisdad espectral de potencia del ruido  
for i in range(len(B)):
    plt.figure('4-'+str(i))
    plt.stem(t,DEP[i])
    plt.title('Densisdad Espectral de potencia del Ruido')
    plt.xlabel('Frecuencia [hz]')
    plt.ylabel('Watts [W]')
##plot de la correlacion del ruido con si mismo
##en 0 esto nos da la potencia media
## y calculo la potencia media integrando la densidad espectral de potencia  
for i in range(len(B)):
    corr[i] = sig.correlate(noise[i], noise[i], mode='same')
    plt.figure('5-'+str(i))
    plt.stem(t,corr[i])
    plt.title('Correlacion Ruido')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Volts [V]')
    pot_tot=np.sum(DEP[i])/N
    print('potencia total=',pot_tot)