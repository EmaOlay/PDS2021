# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:03:37 2021

@author: Ema
"""

#######################################################################################################################
#%% Configuración e inicio de la simulación
#######################################################################################################################
 
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft
 
# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
 
# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 4
N_os = N*over_sampling
 
# Datos del ADC
B = 4 # bits
Vf = 2 # Volts
q = Vf/2**B # Volts
 
# datos del ruido
kn = 1
pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)
 
ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral
 
#######################################################################################################################
#%% Acá arranca la simulación
 #%% Cuentas correspondientes al seno ideal
t = np.linspace(0, 1, N_os,endpoint=False)
t_sampleado= np.linspace(0, (), N)
#senoidal de amplitud 1 y 1 hz de frecuencia
amp=1 #amplitud en volts
freq=1 #fs/N #frecuencia en Hz
fase=0 #radianes
##Armo el seno ideal
analog_sig=amp*np.sin(2 * np.pi * freq * t + fase)
##calculo su potencia en watts
seno_ideal_watts = analog_sig ** 2
##Calculo potencia en dB
seno_ideal_db = 10 * np.log10(seno_ideal_watts)
    #%% Cuentas correspondientes al ruido
##Signal to Noise ratio en dB
SNR_db = 20
##Calculo la media
seno_ideal_avg_watts = np.mean(seno_ideal_watts)
seno_ideal_avg_db = 10 * np.log10(seno_ideal_avg_watts)
##Calculo la potencia del ruido
noise_avg_db = seno_ideal_avg_db - SNR_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
##Armo el ruido con la potencia calculada
noise = np.random.normal(0, np.sqrt(noise_avg_watts), len(seno_ideal_watts))
    #%%Le sumo el ruido al seno ideal
sr = analog_sig + noise

    #%%Cuantizacion
srq=np.round(sr/q)*q
    #%%Calculo de la Densidad espectral de potencia
DEP_srq=np.abs(fft(srq))**2
DEP_sr=np.abs(fft(sr))**2
DEP_analog_sig=np.abs(fft(analog_sig))**2
Noise_q=sr-srq
Noise_q_mean=np.mean(Noise_q)
Noise_mean=np.mean(noise)
#10* np.log10(2* nNn_mean)
#######################################################################################################################
#%% Presentación gráfica de los resultados
plt.close('all')
 
plt.figure(1)
plt.plot(t, srq, lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
plt.plot(t, sr, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
plt.plot(t, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')
 
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

####################################################
 
plt.figure(2)
plt.plot( t, 10* np.log10(2*DEP_srq), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
plt.plot( t, 10* np.log10(2*DEP_analog_sig**2), color='orange', ls='dotted', label='$ s $ (analog)' )
plt.plot( t, 10* np.log10(2*DEP_sr**2), ':g', label='$ s_R = s + n $  (ADC in)' )
plt.plot( t, 10* np.log10(2*noise**2), ':r')
plt.plot( t, 10* np.log10(2*Noise_q**2), ':c')
plt.plot( np.array([ 0, 1 ]), 10* np.log10(2* np.array([Noise_mean, Noise_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* Noise_mean)) )
plt.plot( np.array([ 0, 1 ]), 10* np.log10(2* np.array([Noise_q_mean, Noise_q_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Noise_q_mean)) )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

##################################################

plt.figure(3)
bins = 10
plt.hist(Noise_q, bins=bins)
#Aca agregue el 4* ya que no hice mi array de tiempo "discreto" que es 4 veces menor que t
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, 4*N/bins, 4*N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

 
 