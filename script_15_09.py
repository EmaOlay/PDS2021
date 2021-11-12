# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 21:01:33 2021

@author: Ema
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stats
import matplotlib.patches as mpatches
from scipy.fft import *

plt.close("all")
N = 1000 # muestras
fs= 1000 #Hz

ts = 1/fs # tiempo de muestreo

t=np.arange(0,1,ts)

#senoidal de amplitud 1 y 1 hz de frecuencia
amp=1 #amplitud en volts
freq=1 #fs/N #frecuencia en Hz
fase=0 #radianes
df=fs/N
f=np.arange(0,fs,df/2)

seno_cont0 = amp*np.sin(2 * np.pi * (N/(4)-0)*df * t + fase)
seno_cont1 = amp*np.sin(2 * np.pi * (N/(4)-1)*df * t + fase)
seno_cont2 = amp*np.sin(2 * np.pi * (N/(4)+1)*df * t + fase)
seno_cont3 = amp*np.sin(2 * np.pi * (N/(4)-0.5)*df * t + fase)
seno_cont4 = amp*np.sin(2 * np.pi * (N/(4)+0.5)*df * t + fase)
seno_cont5 = amp*np.sin(2 * np.pi * (N/(4)+0.25)*df * t + fase)
seno_cont6 = amp*np.sin(2 * np.pi * (N/(4)+0.75)*df * t + fase)

dep0=1/N*np.abs(fft(seno_cont0))
dep1=1/N*np.abs(fft(seno_cont1))
dep2=1/N*np.abs(fft(seno_cont2))
dep3=1/N*np.abs(fft(seno_cont3))
dep4=1/N*np.abs(fft(seno_cont4))
dep5=1/N*np.abs(fft(seno_cont5))
dep6=1/N*np.abs(fft(seno_cont6))

bfrec = f <= fs/2

plt.figure('0-')
# plt.plot(f[bfrec],20*np.log10(2*dep0[bfrec]),'x:')
# plt.plot(f[bfrec],20*np.log10(2*dep1[bfrec]),'x:')
# plt.plot(f[bfrec],20*np.log10(2*dep2[bfrec]),'x:')
# plt.plot(f[bfrec],20*np.log10(2*dep3[bfrec]),'x:')
plt.plot(f[0:500],20*np.log10(dep4[0:500]),'x:')
# plt.plot(f[bfrec],20*np.log10(dep5[bfrec]),'x:')
# plt.plot(f[0:500],20*np.log10(dep6[0:500]),'x:')

plt.title('Ploteo')
plt.xlabel('$\Delta f$ [#]')
plt.ylabel('Volts [V]')