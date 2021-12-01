# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:12:02 2021

@author: Ema
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate
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

patron_normal= mat_struct['heartbeat_pattern1']

patron_ventricular= mat_struct['heartbeat_pattern2']

N=len(ecg)

muestras_tot=np.arange(0,N,1)



d_muestras1=200
d_muestras2=300

nn = np.zeros(np.size(qrs_detections))

#%%Verifico haber levantado bien la data
# plt.figure(figura)
# plt.plot(muestras_tot,ecg)
# figura+=1

#%%Verifico encontrar un latido
# plt.figure(figura)
# plt.plot(muestras_tot,ecg)
# figura+=1
# pos=qrs_detections[1000]

# plt.xlim(pos-d_muestras1,pos+d_muestras2)
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

#%%Busco agrandar mis dos patrones para correlacionar en cada deteccion
y=patron_normal[:,0]
x = np.arange(y.size)

# Interpolate the data using a cubic spline to "new_length" samples
Nuev_largo = d_muestras1+d_muestras2
Nuevo_x = np.linspace(x.min(), x.max(), Nuev_largo)
Nuevo_patron_normal = sp.interpolate.interp1d(x, y, kind='cubic')(Nuevo_x)

# Plot del antes y despues
# plt.figure(10)
# plt.subplot(2,1,1)
# plt.plot(patron_normal, 'bo-')
# plt.title('Using 1D Cubic Spline Interpolation')

# plt.subplot(2,1,2)
# plt.plot(Nuevo_x, Nuevo_patron_normal, 'ro-')

#####################
##Idem Ventricular
#####################
y=patron_ventricular[:,0]
x = np.arange(y.size)

# Interpolate the data using a cubic spline to "new_length" samples
Nuev_largo = d_muestras1+d_muestras2
Nuevo_x = np.linspace(x.min(), x.max(), Nuev_largo)
Nuevo_patron_ventricular = sp.interpolate.interp1d(x, y, kind='cubic')(Nuevo_x)

# Plot del antes y despues
# plt.figure(11)
# plt.subplot(2,1,1)
# plt.plot(patron_ventricular, 'bo-')
# plt.title('Using 1D Cubic Spline Interpolation')

# plt.subplot(2,1,2)
# plt.plot(Nuevo_x, Nuevo_patron_ventricular, 'ro-')
#%%Busco separar en grupos para correlacionar

for i in range(1903):
    
    # nn[i]=np.cov(Nuevo_patron_normal,array_latidos[:,i])
    nn[i]=np.corrcoef(array_latidos[:,i],Nuevo_patron_normal)[0,1]
 



plt.figure(11)
bool_vector = np.abs(nn) > 0.7*nn[np.argmax(np.abs(nn))]##Busco fuerte correlacion

plt.plot(nn[bool_vector], 'bo')
plt.title('cuantos puntos encontre')
plt.figure(12)
plt.hist(nn[bool_vector])
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

plt.plot(f_welch,10*np.log10(Pxx_den/np.argmax(Pxx_den,axis=0)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('$Potencia (dB)$')
#plt.ylim(-10,30)
plt.xlim(0,35)

#%% Con lo anterior puedo estimar que tengo 2 señales aunque 1 de ellas tiene menos energia que la otra...
#Por lo pronto armo el filtro para la que tiene mas energia
#####Parametros para armar el filtro
nyq=fs/2
lowstop=0.1
lowcut = 1
highcut = 35
highstop=45
low_stop=lowstop/nyq
low_pass=lowcut/nyq
high_pass=highcut/nyq
high_stop=highstop/nyq
gpass = 1
gstop = 30
#####Armo el primer filtro IIR
system=sig.iirdesign(wp=[low_pass,high_pass],
                     ws=[low_stop,high_stop],gpass=gpass,gstop=gstop,analog=False,
                     ftype='butter',output='sos')
##Muestro el filtro
plt.figure(figura)
figura+=1
w, h = sig.sosfreqz(system,fs=fs,worN=2000)
##tiene pinta pero no se...
plt.plot(w, 20 * np.log10(abs(h)))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.xlim(0,100)

#####filtro la senal contra el ecg completo
y= sig.sosfiltfilt(system, ecg,axis=0,padtype='odd',padlen=None)
#Imprimo el resultado
plt.figure(figura)
figura+=1
plt.plot(y,'b-')
plt.plot(ecg,'r-')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')

#%%Armo el segundo filtro IIR
system=sig.iirdesign(wp=[low_pass,high_pass],
                     ws=[low_stop,high_stop],gpass=gpass,gstop=gstop,analog=False,
                     ftype='cheby1',output='sos')
##Muestro el filtro
plt.figure(figura)
figura+=1
w, h = sig.sosfreqz(system,fs=fs,worN=2000)
##tiene pinta pero no se...
plt.plot(w, 20 * np.log10(abs(h)))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.xlim(0,100)

#filtro la senal contra el ecg completo
y= sig.sosfiltfilt(system, ecg,axis=0,padtype='odd',padlen=None)
#Imprimo el resultado
plt.figure(figura)
figura+=1
plt.plot(y,'b-')
plt.plot(ecg,'r-')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')

#%%Armo el primer filtro FIR

numtaps=2000##debe ser par para que el numtaps en al funcion sea impar

b=sig.firwin(numtaps=numtaps+1, cutoff=[lowcut, highcut], pass_zero=False,fs=fs)
##Muestro el filtro
plt.figure(figura)
figura+=1
w,h=sig.freqz(b=b, a=1, worN=2000, whole=False, plot=None, fs=fs, include_nyquist=False)
plt.plot(w, 20 * np.log10(abs(h)))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.xlim(0,100)

#%%Armo el segundo filtro FIR
numtaps=1500##debe ser par para que el numtaps en al funcion sea impar

bands=(0,0.1,lowcut,highcut,highstop,nyq)
desired=(0,0,1,1,0,0)
##b=sig.remez(numtaps=numtaps+1, bands=bands, desired=desired, type='bandpass', fs=fs)

##Muestro el filtro
plt.figure(figura)
figura+=1
w,h=sig.freqz(b=b, a=1, worN=2000, whole=False, plot=None, fs=fs, include_nyquist=False)
plt.plot(w, 20 * np.log10(abs(h)))

plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.xlim(0,100)
