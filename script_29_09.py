# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:33:22 2021

@author: Ema
"""

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
K1=N/4+0.25
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

#%% Potencia de mi señal
#Este es el valor de la energia en el tiempo, se debe repetir en frecuencia
#por Parseval asique lo puedo usar para normalizar o comparar que este todo bien
#No olvidar el tema de la cantidad de muestras
E0=1/N*sum(np.abs(seno_cont0)**2)
print("Potencia del seno cont0=",E0)
E1=1/N*sum(np.abs(seno_cont1)**2)
print("Potencia del seno cont1=",E1)
E2=1/N*sum(np.abs(seno_cont2)**2)
print("Potencia del seno cont2=",E2)
#%%Normalizo la señal
sin_nom0=seno_cont0/np.sqrt(E0)
print("Potencia normalizada 1=",np.var(sin_nom0))
sin_nom1=seno_cont1/np.sqrt(E1)
print("Potencia normalizada 2=",np.var(sin_nom1))
sin_nom2=seno_cont2/np.sqrt(E2)
print("Potencia normalizada 3=",np.var(sin_nom2))
#%% Calculo la densidad espectral de potencia
#Si integro estas funciones densisdad tendrian que ser iguales a sus correspondientes
#Energias en tiempo
DEP0=np.abs(1/N*fft(sin_nom0))**2
DEP1=np.abs(1/N*fft(sin_nom1))**2
DEP2=np.abs(1/N*fft(sin_nom2))**2

DEP0_int=sum(DEP0)
print("Potencia en tiempo=",E0,"Potencia en frec=",DEP0_int,"para señal 0")
DEP1_int=sum(DEP1)
print("Potencia en tiempo=",E1,"Potencia en frec=",DEP1_int,"para señal 1")
DEP2_int=sum(DEP2)
print("Potencia en tiempo=",E2,"Potencia en frec=",DEP2_int,"para señal 2")
#%% Muestro las DEP o PSD power spectral density

plt.figure(1)
#Como muestro la mitad de las frecuencias hago el doble la DEP
plt.plot(f, 2*DEP0)
plt.title("DEP0")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
plt.xlim(0, fs/2)

plt.figure(2)
#Como muestro la mitad de las frecuencias hago el doble la DEP
plt.plot(f, 2*DEP1)
plt.title("DEP1")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
plt.xlim(0, fs/2)

plt.figure(3)
#Como muestro la mitad de las frecuencias hago el doble la DEP
plt.plot(f, 2*DEP2)
plt.title("DEP2")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
plt.xlim(0, fs/2)
#%%Armo los nuevos vectores f
N_prima=9*N
df_prima=fs/N_prima
f_prima=np.arange(0,fs,df_prima)

#%%Aplico el zero padding solicitado en el bonus
#np.abs(1/N*fft(sin_nom0))**2
DEP0_prima=np.abs(1/N_prima*fft(seno_cont0))**2
DEP1_prima=np.abs(1/N_prima*fft(seno_cont1))**2
DEP2_prima=np.abs(1/N_prima*fft(seno_cont2))**2
#%% Muestro las DEP o PSD power spectral density
plt.figure(1)

plt.stem(f_prima, DEP0_prima,linefmt='red', markerfmt='D')
plt.title("DEP0")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
#plt.xlim(240, 260)
print("Potencia paddeada=",sum(DEP0_prima))

plt.figure(2)
plt.stem(f_prima, DEP1_prima,linefmt='red', markerfmt='D')
plt.title("DEP1")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
#plt.xlim(240, 260)
print("Potencia paddeada=",sum(DEP1_prima))

plt.figure(3)
plt.stem(f_prima, DEP2_prima,linefmt='red', markerfmt='D')
plt.title("DEP2")
plt.ylabel("Potencia[W]")
plt.xlabel("frec[Hz]")
#plt.xlim(240, 260)
print("Potencia paddeada=",sum(DEP2_prima))