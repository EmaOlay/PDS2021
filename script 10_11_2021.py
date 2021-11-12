# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 20:46:14 2021

@author: Ema
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io as sio
from scipy.fft import fft, fftshift
import matplotlib.patches as mpatches
from pandas import DataFrame
from IPython.display import HTML
#######################################################################################################################
#%% En caso de usar spectrum
#######################################################################################################################
import spectrum
from spectrum.datasets import marple_data
from pylab import legend, ylim
#######################################################################################################################
#%% Inicio de la simulación
#######################################################################################################################
plt.close('all')

#sio.whosmat('ECG_TP4.mat')
mat_struct= sio.loadmat('ECG_TP4.mat')

fs= 1000 #Hz
figura=0

ecg= mat_struct['ecg_lead']

qrs_detections = mat_struct['qrs_detections']

N=len(ecg)

muestras_tot=np.arange(0,N,1)

d_muestras1=200
d_muestras2=350


#%%Verifico haber levantado bien la data
plt.figure(figura)
plt.plot(muestras_tot,ecg)
figura+=1

figura_a=10
figura_b=11

#%%Armo mi matriz de latidos con el delta muestras para adelante y atras
#la matriz contiene tantos elementos coomo qrs_detections de intervalo 
#d_muestras1+d_muestras2
latidos_matrix= [ (ecg[int(i-d_muestras1):int(i+d_muestras2)]) for i in qrs_detections ]
##Con esto me aseguro que quede como array y no como lista de arrays
array_latidos=np.hstack(latidos_matrix)
#Los latidos estan a distintas alturas entonces sincronizo en y
#Tome la decision de restar la media de todo el experimento lo que en la mayoria
#de los casos deberia ser correcto
array_latidos=array_latidos[:,:50] - np.mean(array_latidos[:,:50],axis=0)

array_latidos_padded=np.pad(array_latidos,pad_width=((2000,2000),(0,0)),mode='constant')

array_latidos_padded_prom=np.median(array_latidos_padded,axis=1)

plt.figure(figura)
figura+=1
plt.plot(array_latidos_padded)
plt.plot(array_latidos_padded_prom,'--',lw=2)


#%% Calculo la Densidad espectral de potencia PSD para todos(Power Spectral Density)
N_array=len(array_latidos_padded)
#Metodo de Welch paddeado para hacerlo mas suave
#Pruebo ventanas: bartlett, hanning, blackman, flattop
#Armar el solapamiento correspondiente para detectar los latidos
f_welch,Pxx_den = sig.welch(array_latidos_padded,fs=fs,nperseg=N_array/2,axis=0)

#Imprimo el resultado
plt.figure(figura_a)
figura+=1
#plt.semilogy(f_welch,Pxx_den)
plt.plot(f_welch,Pxx_den)
plt.xlabel('Frequency (Hz)')
plt.ylabel('$V^2/Hz$')
# plt.xlim(0,35)

#Imprimo el resultado
plt.figure(figura_b)
figura+=1

plt.plot(f_welch,10*np.log10(Pxx_den/Pxx_den.argmax()))
plt.xlabel('Frequency (Hz)')
plt.ylabel('$Potencia (dB)$')
# plt.ylim(-10,30)
# plt.xlim(0,35)

#%% Calculo la Densidad espectral de potencia PSD para la mediana(Power Spectral Density)
N_array=len(array_latidos_padded_prom)
#Metodo de Welch paddeado para hacerlo mas suave
#Pruebo ventanas: bartlett, hanning, blackman, flattop
#Armar el solapamiento correspondiente para detectar los latidos
f_welch,Pxx_den_prom = sig.welch(array_latidos_padded_prom,fs=fs,nperseg=N_array/2,axis=0)

#Imprimo el resultado
plt.figure(figura_a)
figura+=1
#plt.semilogy(f_welch,Pxx_den)
plt.plot(f_welch,Pxx_den_prom,'kx-',lw=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('$V^2/Hz$')
# plt.xlim(0,35)

plt.plot(f_welch,np.median(Pxx_den,axis=1),'ro-',lw=2)

#%%Integro sobre el PSD de la mediana

Pot_tot=np.cumsum(Pxx_den_prom)/np.sum(Pxx_den_prom)
#Revisar filtrado de fase 0 grabacion 21:30
porcetaje=0.99

##Probar todo anterior con partes de la ecg aprox muestra 690.000