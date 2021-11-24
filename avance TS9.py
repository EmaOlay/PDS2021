# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:12:02 2021

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

patron1= mat_struct['heartbeat_pattern1']

patron2= mat_struct['heartbeat_pattern2']

N=len(ecg)

muestras_tot=np.arange(0,N,1)


#%%Verifico haber levantado bien la data
plt.figure(figura)
plt.plot(muestras_tot,ecg)
figura+=1


#%%Verifico encontrar un latido
plt.figure(figura)
plt.plot(muestras_tot,ecg)
figura+=1
pos=qrs_detections[1000]
d_muestras1=200
d_muestras2=300
plt.xlim(pos-d_muestras1,pos+d_muestras2)
#%%Armo mi matriz de latidos con el delta muestras para adelante y atras
#la matriz contiene tantos elementos coomo qrs_detections de intervalo 
#d_muestras1+d_muestras2
latidos_matrix= [ (ecg[int(i-d_muestras1):int(i+d_muestras2)]) for i in qrs_detections ]
##Con esto me aseguro que quede como array y no como lista de arrays
array_latidos=np.hstack(latidos_matrix)
#Los latidos estan a distintas alturas entonces sincronizo en y
#Tome la decision de restar la media de todo el experimento lo que en la mayoria
#de los casos deberia ser correcto
array_latidos=array_latidos - np.mean(array_latidos,axis=0)


#%%Busco imprimir los latidos superpuestos
#Armo un vector para el eje x de esta representacion
muestras_latido=np.arange(0,d_muestras1+d_muestras2,1)

#Verifico todos los latidos uno arriba del otro
plt.figure(figura)
pico_array_latidos=array_latidos.argmax()
plt.plot(muestras_latido,array_latidos/pico_array_latidos)
figura+=1

#%%Busco verificar la forma en promedio que tuvieron todos los latidos
#Hago la media de cada una de las 500 muestras, de las 1903 veces que encontre un latido
latido_promedio=np.mean(array_latidos,axis=1)

plt.figure(figura)
plt.plot(muestras_latido,latido_promedio)
figura+=1

#%% Calculo la Densidad espectral de potencia PSD(Power Spectral Density)
# N_array=len(array_latidos)
# #Metodo de Welch paddeado para hacerlo mas suave
# #Pruebo ventanas: bartlett, hanning, blackman, flattop
# f_welch,Pxx_den = sig.welch(array_latidos,fs=fs,nperseg=N_array/5,nfft=5*N_array,axis=0)

# #Imprimo el resultado
# plt.figure(figura)
# figura+=1
# #plt.semilogy(f_welch,Pxx_den)
# plt.plot(f_welch,Pxx_den)
# plt.xlim(0,35)

#%% Calculo la Densidad espectral de potencia PSD(Power Spectral Density)
N_array=len(array_latidos)
#Metodo de Welch paddeado para hacerlo mas suave
#Pruebo ventanas: bartlett, hanning, blackman, flattop
#Armar el solapamiento correspondiente para detectar los latidos
f_welch,Pxx_den = sig.welch(array_latidos,fs=fs,nperseg=N_array/2,window='bartlett',nfft=10*N_array,axis=0)

#Imprimo el resultado
plt.figure(figura)
figura+=1
#plt.semilogy(f_welch,Pxx_den)
plt.plot(f_welch,Pxx_den)
plt.xlabel('Frequency (Hz)')
plt.ylabel('$V^2/Hz$')
plt.xlim(0,35)

#Imprimo el resultado
plt.figure(figura)
figura+=1

plt.plot(f_welch,10*np.log10(Pxx_den/Pxx_den.argmax()))
plt.xlabel('Frequency (Hz)')
plt.ylabel('$Potencia (dB)$')
#plt.ylim(-10,30)
plt.xlim(0,35)

#%% Con lo anterior puedo estimar que tengo 2 señales aunque 1 de ellas tiene menos energia que la otra...
#Por lo pronto armo el filtro para la que tiene mas energia
#####Parametros para armar el filtro
nyq=fs/2
lowcut = 5
highcut = 15
low=lowcut/nyq
high=highcut/nyq
#####Armo el filtro
b,a=sig.butter(N=6, Wn=[low,high], btype='band')
##Muestro el filtro
plt.figure(figura)
figura+=1
w, h = sig.freqz(b, a)
##tiene pinta pero no se...
plt.plot((nyq / np.pi) * w, abs(h))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.xlim(0,35)

#####filtro la senal
y= sig.lfilter(b, a, array_latidos,axis=1)
#Imprimo el resultado
plt.figure(figura)
figura+=1
plt.plot(muestras_latido, y)
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
#### Parece que filtre el ventricular
