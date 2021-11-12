# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:52:46 2021

@author: Ema
Bilbio para la res de esta tarea Holton pag 888
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
import matplotlib.patches as mpatches
from scipy.fft import *

#%% Set-UP
plt.close("all")
N = 1000 # muestras
fs= 1000 #Hz

ts = 1/fs # tiempo de muestreo

t=np.arange(0,1,ts)

K0=N/4
K1=N/4+0.025
K2=N/4+0.5
#%% Config de mi seno y vector de frecuencia
#senoidal de amplitud 1
amp=1 #amplitud en volts
freq0=K0*fs/N #frecuencia en Hz

fase=0 #radianes
df=fs/N
f=np.arange(0,fs,df)
bfrec = f <= fs/2
seno_cont0 = amp*np.sin(2 * np.pi * freq0 * t + fase)

#%%Config del seno 2
freq1=K1*fs/N #frecuencia en Hz
seno_cont1 = np.sqrt(amp)*np.sin(2 * np.pi * freq1 * t + fase)

#%%Config del seno 3
freq2=K2*fs/N #frecuencia en Hz
seno_cont2 = amp*np.sin(2 * np.pi * freq2 * t + fase)
#%% Energia de mi señal
#Este es el valor de la energia en el tiempo, se debe repetir en frecuencia
#por Parseval asique lo puedo usar para normalizar o comparar que este todo bien
#No olvidar el tema de la cantidad de muestras
E0=sum(np.abs(seno_cont0)**2)
E1=sum(np.abs(seno_cont1)**2)
E2=sum(np.abs(seno_cont2)**2)

#%% Calculo la densidad espectral de potencia
#Si integro estas funciones densisdad tendrian que ser iguales a sus correspondientes
#Energias en tiempo
DEP0=np.abs(fft(seno_cont0))**2
DEP1=np.abs(fft(seno_cont1))**2
DEP2=np.abs(fft(seno_cont2))**2

DEP0_int=sum(DEP0)/N
DEP1_int=sum(DEP1)/N
DEP2_int=sum(DEP2)/N
#%% Muestro las DEP o PSD power spectral density
plt.figure(1)

plt.plot(f, DEP0/(E0*N))
plt.title("DEP0")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
plt.xlim(0, fs/2)

plt.figure(2)
plt.plot(f, DEP1/(E1*N))
plt.title("DEP1")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
plt.xlim(0, fs/2)

plt.figure(3)

plt.plot(f, DEP2/(E2*N))
plt.title("DEP2")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
plt.xlim(0, fs/2)
##Al visualizar las 3 DEP puedo ver que ese pequeño desfasaje en la delta f
##Hace que mi potencia ya no se concentre en el un punto sino que se desparrama
##A otras frecuencias vecinas
#%% Config de la window
window = sig.windows.boxcar(len(t))

####Verificacion visual de la window en t
# plt.figure(0)
# plt.plot(t,window)
# plt.title("Boxcar window")
# plt.ylabel("Amplitud")
# plt.xlabel("Tiempo")

#%%Preparo las salidas para despues plotear
    #%%caso 1 no se distingue la sincoide
    
# fft_w=np.abs(fft(window)/len(window))
# freqs_1 = fftfreq(fft_w.size, 0.1)
# plt.plot(freqs_1, fft_w)
# plt.xlim(0, 0.5)


    #%%caso 2 se distingue la sincoide
# w_fft = fft(window, n=5000)
# freqs = fftfreq(w_fft.size, 0.1)
# ##Normalizo por su valor pico
# w_fft_db=20*np.log10(np.abs(w_fft) / np.abs(w_fft).max())
# plt.figure(1)
# plt.plot(freqs, w_fft_db,'--r')
# plt.xlim(0, 0.1)
# plt.ylim(-30,0)


# #%%fft del seno
# fft_x=np.abs(fft(seno_cont0)/len(seno_cont0))

#%%convol
# convol=sig.convolve(fft_x,fft_w)
#%%Plots

# plt.figure(1)

# plt.plot(f, 20*np.log10(np.abs(fft_w) / np.abs(fft_w).max()))
# plt.title("Boxcar window fft")
# plt.ylabel("Amplitud")
# plt.xlabel("frec")


# plt.figure(2)

# plt.plot(f[0:500],fft_x[0:500])
# plt.title("seno fft")
# plt.ylabel("Amplitud")
# plt.xlabel("frec")