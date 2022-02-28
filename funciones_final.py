# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:47:22 2022

@author: Ema
"""
#%% Imports
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import scipy as sp
import scipy.signal as sig
from scipy.fft import fft, fftfreq, fftshift

#%% Un par de funciones

#elimino algo del array que no me gusta

def quito_dato(array,dato):
    array = [x for x in array if x != dato]
    return array


# Normalización

def normalizacion(signal):
    maximo = np.amax(np.abs(signal))
    return signal/maximo

#funcion que filtra y calcula el ritmo para el ecg que le pase.
#No esta pensada para mostrar los graficos pero no me costaba mucho agregar la posibilidad.
#ecg es la se;al a partir de la cual vamos a trabajar.
#resp_ref es la se;al objetivo a obtener contra lo cual comparamos.
#El parametro print nos indica distintos graficos que querramos ver.
def calculo_ritmo_respiratorio(ecg,resp_ref,fs=0,p_print='Nada'):
    ##Verifico recibir los parametros correctos
    if fs<=0:
        print('No se ingreso una fs valida')
        return
    if p_print == 'Nada':  
        pass
        #print('Elegiste no imprimir nada')
######Hago mi interpolacion
    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs)
    
    detecciones=rpeaks['ECG_R_Peaks']
    
    muestras=ecg.size
    ##el tiempo empieza en 0 termina en muestras/fs y tiene num puntos en el medio
    t = np.linspace(0, muestras+detecciones[0],num=detecciones.size,endpoint=True)
    ##f2 es mi funcion que interpola para t y los puntos R del ecg con una cubica
    f2 = sp.interpolate.interp1d(t, ecg[detecciones], kind='cubic')
    ##el tiempo empieza en 0 termina en muestras/fs y tiene num puntos en el medio
    tnew = np.linspace(0, muestras, num=muestras, endpoint=True)
    #Creo mi extraccion con la funcion interpoladora
    extraccion=f2(tnew)
    #Desfaso respecto del primer pico R que encuentro
    extraccion=f2(tnew + rpeaks['ECG_R_Peaks'][0])
    #Le quito la media que le genera la funcion interpoladora para ponerlo lo mas cerca de cero posible
    extraccion= extraccion - np.mean(extraccion)
    
#######Empiezo a filtrar
    #Plantilla pasa altos
    
    f_paso=4/60 #respiraciones por segundo 0.0666
    f_stop=0.006
    #Ganancias en las bandas
    gpass = 0.5
    gstop = 28
    sos_HP=sig.iirdesign(wp=f_paso*2*np.pi,
                         ws=f_stop*2*np.pi,gpass=gpass,gstop=gstop,analog=False,
                         ftype='cheby1',output='sos',fs=fs)
    #####filtro mi extraccion para quitar frecuencias que se hayan agregado
    y_HP= sig.sosfiltfilt(sos_HP, extraccion)
    y_HP_ref= sig.sosfiltfilt(sos_HP, resp_ref)
    
    #Plantilla pasa bajos

    f_paso=64/60  #respiraciones por segundo 1.066
    f_stop=1.5
    
    gpass = 0.5
    gstop = 30
    sos_LP=sig.iirdesign(wp=f_paso*2*np.pi,
                         ws=f_stop*2*np.pi,gpass=gpass,gstop=gstop,analog=False,
                         ftype='cheby1',output='sos',fs=fs)
    #####filtro la senal contra la salida anterior
    y_LP= sig.sosfiltfilt(sos_LP, y_HP)
    y_LP_ref= sig.sosfiltfilt(sos_LP, y_HP_ref)
    
    #Plantilla Notch

    b,a=sig.iirnotch(w0=50,Q=1.5,fs=fs)
    #####filtro la senal contra la salida anterior
    y= sig.filtfilt(b,a, y_LP)
    y_ref= sig.filtfilt(b,a, y_LP_ref)
#######Fin del Filtrado

#######Calculo del ancho de banda a partir del periodograma de Welch
    npad=muestras*10
    f_welch_ref_o,Pxx_ref_o = sig.welch(resp_ref,fs=fs, nperseg=y_ref.size/4,window='bartlett',nfft=npad)
    f_welch_ref,Pxx_ref = sig.welch(y_ref,fs=fs, nperseg=y_ref.size/4,window='bartlett',nfft=npad)
    f_welch_y,Pxx_y = sig.welch(y,fs=fs, nperseg=y.size/4,window='bartlett',nfft=npad)
#######FIN Calculo de Welch

#######Calculo de la FFT
    # sample spacing
    T = 1.0 / fs
    fft_y=fft(y,n=npad)
    fft_ref=fft(y_ref,n=npad)
    fft_ref_o=fft(resp_ref,n=npad)
    #Agrego el metodo del neurokit
    van = nk.ecg_rsp(ecg, sampling_rate=fs)
    fft_van=fft(van,n=npad)
    #Eje de frecuencia
    xf = fftfreq(npad, T)[:npad//2]
#######FIN Calculo de la FFT

#######Imprimo los resultados pedidos
    if p_print == 'Interpolacion':
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(normalizacion(extraccion),'b-',label='Mi extraccion')
        plt.plot(normalizacion(resp_ref),'r-',label='Referencia')
        van = nk.ecg_rsp(ecg, sampling_rate=fs)
        plt.plot(normalizacion(van),'g-',label='Van Gent et al.')        
        plt.title('Respiraciones')
        plt.grid(True)
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(normalizacion(ecg))
        # Visualize R-peaks in ECG signal
        for xc in detecciones:
            plt.axvline(x=xc, color='r', linestyle=':')
        plt.title('Detecciones de picos R')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
    if p_print == 'Plantilla HP':
        ##Muestro el filtro
        plt.figure(2)
        w, h = sig.sosfreqz(sos_HP,fs=fs)    
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.plot(w, 20 * np.log10(np.abs(h)))
        print('Las plantillas no les gusta mucho frecuencias tan bajas y suelen hacer cosas raras')
    if p_print == 'Plantilla LP':
        ##Muestro el filtro
        plt.figure(3)
        w, h = sig.sosfreqz(sos_LP,fs=fs)
        ##tiene pinta
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.xlim(-0.2,10)
        plt.ylim(-10,2)
        print('Las plantillas no les gusta mucho frecuencias tan bajas y suelen hacer cosas raras')
    if p_print == 'Plantilla Notch':
        ##Muestro el filtro
        plt.figure(4)
        w, h = sig.freqz(b,a,fs=fs)
        ##Se podria mejorar el notch pero meh
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.xlim(0,100)
    if p_print == 'Salida HP':
        #Imprimo el resultado
        plt.figure(5)
        plt.subplot(3,1,1)
        plt.plot(normalizacion(y_HP),'b-',label='Mi extraccion')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.title('Salida del Filtro Pasa Altos')
        plt.grid(True)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(normalizacion(y_HP_ref),'r-',label='Referencia filtrada')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.grid(True)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(normalizacion(resp_ref),'k-',label='Referencia')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.grid(True)
        plt.legend()
    if p_print == 'Salida LP':
        #Imprimo el resultado
        plt.figure(6)
        plt.subplot(3,1,1)
        plt.plot(normalizacion(y_LP),'b-',label='Mi extraccion')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.title('Salida del Filtro Pasa Bajos')
        plt.grid(True)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(normalizacion(y_LP_ref),'r-',label='Referencia filtrada')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.grid(True)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(normalizacion(resp_ref),'k-',label='Referencia')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.grid(True)
        plt.legend()
        print('La salida del LP es despues de filtrar por el HP')
    if p_print == 'Salida Notch':
        #Imprimo el resultado
        plt.figure(7)
        plt.subplot(3,1,1)
        plt.plot(normalizacion(y),'b-',label='Mi extraccion')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.title('Salida del Filtro Notch')
        plt.grid(True)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(normalizacion(y_ref),'r-',label='Referencia filtrada')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.grid(True)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(normalizacion(resp_ref),'k-',label='Referencia')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.grid(True)
        plt.legend()
    if p_print == 'BW_1':
        plt.figure(8)
        plt.subplot(1,3,1)
        plt.plot(f_welch_y,(Pxx_y/np.amax(Pxx_y)),'b-',label='Mi extraccion')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$PDS$[$V^{2}/Hz$]')
        plt.ylim(-0.2,1.1)
        plt.xlim(0,1)
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(f_welch_ref,(Pxx_ref/np.amax(Pxx_ref)),'r-',label='Ref filtrada')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$PDS$[$V^{2}/Hz$]')
        plt.ylim(-0.2,1.1)
        plt.xlim(0,1)
        plt.legend()
        plt.subplot(1,3,3)
        plt.plot(f_welch_ref,(Pxx_ref/np.amax(Pxx_ref)),'k-',label='Ref Original')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('$PDS$[$V^{2}/Hz$]')
        plt.ylim(-0.2,1.1)
        plt.xlim(0,1)
        plt.legend()
        
    if p_print == 'BW_2': 
        corte_energia = 0.95
        Pot_ref = np.cumsum(Pxx_ref)/np.sum(Pxx_ref)
        Pot = np.cumsum(Pxx_y)/np.sum(Pxx_y)
        corte_ref = np.where(Pot_ref >corte_energia)[0][0]
        corte = np.where(Pot >corte_energia)[0][0]
        plt.figure(9)
        plt.subplot(1,2,1)
        plt.plot(f_welch_ref_o,Pxx_ref_o/np.amax(Pxx_ref_o), 'k')
        plt.fill_between(f_welch_ref_o, 0, Pxx_ref_o/np.amax(Pxx_ref_o), where = f_welch_ref_o < f_welch_ref_o[corte_ref], color='red')
        plt.title('Ancho de banda donde se concentra el {:3.0f}% de la energia en la referencia/ mi extraccion'.format(corte_energia*100))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('PSD [$V^{2}/Hz$]')
        plt.xlim(0,1.2)
        
        plt.annotate(   "BW_n = {:3.1f} Hz".format(f_welch_ref_o[corte_ref]),
                        xy=(f_welch_ref_o[corte_ref], Pxx_ref_o[corte_ref]/np.amax(Pxx_ref_o)),
                        xytext=(-20,20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle='->')
        )
        
        plt.subplot(1,2,2)
        plt.plot(f_welch_y,Pxx_y/np.amax(Pxx_y), 'k')
        plt.fill_between(f_welch_y, 0, Pxx_y/np.amax(Pxx_y), where = f_welch_y < f_welch_y[corte], color='blue')
        #plt.title('Ancho de banda donde se concentra el {:3.0f}% de la energia en mi extraccion'.format(corte_energia*100))
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('PSD [$V^{2}/Hz$]')
        plt.xlim(0,1.2)
        
        plt.annotate(   "BW_n = {:3.1f} Hz".format(f_welch_y[corte]),
                        xy=(f_welch_y[corte], Pxx_y[corte]/np.amax(Pxx_y)),
                        xytext=(-20,20),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle='->')
        )
    if p_print == 'FFT':
        plt.figure(10)
        plt.subplot(3,1,1)
        plt.plot(xf, normalizacion(np.abs(fft_y[0:npad//2])),'b-',label='Mi extraccion')
        plt.axvline(xf[np.argmax(np.abs(fft_y[0:npad//2]))], color='r', linestyle=':')
        plt.title('FFT de las Señales')
        plt.xlabel('Frecuencia [Hz]')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        
        plt.xlim(0,1.4)
        plt.legend()
        plt.subplot(3,1,2)
        plt.plot(xf, normalizacion(np.abs(fft_ref[0:npad//2])),'r-',label='Ref filtrada')
        plt.axvline(xf[np.argmax(np.abs(fft_ref[0:npad//2]))], color='r', linestyle=':')
        plt.xlabel('Frecuencia [Hz]')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        
        plt.xlim(0,1.4)
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(xf,normalizacion(np.abs(fft_ref_o[0:npad//2])),'k-',label='Ref Original')
        plt.axvline(xf[np.argmax(np.abs(fft_ref_o[0:npad//2]))], color='r', linestyle=':')
        plt.xlabel('Frecuencia [Hz]')
        #Con estas dos sentencias escondo los ticks de los ejes
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlim(0,1.4)
        plt.legend()
        plt.show()
    
    ritmo_extraccion = xf[np.argmax(np.abs(fft_y[0:npad//2]))]
    ritmo_original = xf[np.argmax(np.abs(fft_ref_o[0:npad//2]))]
    ritmo_van = xf[np.argmax(np.abs(fft_van[0:npad//2]))]
    # input("Press Enter to continue...")
    return ritmo_extraccion,ritmo_original,ritmo_van