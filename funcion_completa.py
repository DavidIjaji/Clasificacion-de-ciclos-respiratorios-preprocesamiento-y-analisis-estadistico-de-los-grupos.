# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:00:30 2020

@author: pc
"""
import librosa;
import librosa.display;
import matplotlib.pyplot as plt;
import numpy as np;
from linearFIR import filter_design, mfreqz;
import scipy.signal as signal;
import pywt;
from tkinter import Tk;
from tkinter.filedialog import askopenfilename;
import glob
import pandas as pd

def Carga_Filt(y,sr): 
    fs = sr;
    #design
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 1000, revfilt = 0);
    #mfreqz(lowpass,1,order, fs/2);
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);
    #mfreqz(highpass,1,order, fs/2);
    
    y_hp = signal.filtfilt(highpass,1,y);
    y_bp = signal.filtfilt(lowpass,1,y_hp);
    
    y_bp = np.asfortranarray(y_bp);
    return (y_bp)


def Filtrado_wavelet(data):
    LL = int(np.floor(np.log2(data.shape[0])));

    coeff = pywt.wavedec(data, 'db6', level=LL);
    
    thr = thselect(coeff);
    coeff_t = wthresh(coeff,thr);
    
    x_rec = pywt.waverec( coeff_t, 'db6');
    
    x_rec = x_rec[0:data.shape[0]];
#    plt.plot(data,label='Original')
#    plt.plot(x_rec,label='Umbralizada por Wavelet')
    x_filt = np.squeeze(data - x_rec);
#    plt.plot(x_filt,label='Original - Umbralizada')
#    plt.legend()
    return(x_filt)
    
def wthresh(coeff,thr):
    y   = list();
    s = wnoisest(coeff);
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
    return y;
    
def thselect(signal):
    Num_samples = 0;
    for i in range(0,len(signal)):
        Num_samples = Num_samples + signal[i].shape[0];
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wnoisest(coeff):
    stdc = np.zeros((len(coeff),1));
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
    return stdc;

def Informacion_sujeto(texto,audio,fs):
#    texto = np.loadtxt(ruta_txt)
#    audio,fs = librosa.load(ruta_wav)
    #audio se reescribe ya que se ha filtrado previamente solo se necesita saber
    # su fs
    Ts=1/fs
    info={}
    i=1
    for linea in texto:
        info['ciclo'+str(i)]=[audio[int(linea[0]/Ts):int(linea[1]/Ts)],int(linea[2]),int(linea[3])]
        i=i+1
    return(info)
    
def Indices(ciclo,fs):
    varianza = np.round(np.var(ciclo),10)
    rango = np.round((np.absolute(np.max(ciclo)-np.min(ciclo))),5)
#    tam_ventana=int((100*ciclo.shape[0])/800)
    tam_ventana=100
    SMA=np.round(max(mov_avg(ciclo,tam_ventana)),6)
    f, Pxx_spec = signal.welch(ciclo, fs, nperseg=1024, scaling='spectrum')
    promedio_espectral=np.round((sum(Pxx_spec)/len(Pxx_spec)),11)
    return(varianza,rango,SMA,promedio_espectral)
    
def mov_avg(x, w):
    for m in range(len(x)-(w-1)):
        yield sum(np.ones(w) * x[m:m+w]) / w 
    
def Funcion_madre(ruta_archivos):
    #se generan listas de todos los archivos txt y wav de la carpeta
    archivos_texto = glob.glob(ruta_archivos + '\*.txt')
    archivos_audio = glob.glob(ruta_archivos + '\*.wav') 
    VARIANZAS   = np.array([])
    RANGOS      = np.array([])
    SMAS        = np.array([])
    PROMEDIOS   = np.array([])
    ESTADOS     = np.array([])
#    ESTERTORES = np.array([])
#    SILIBANCIAS= np.array([])
    for i in np.arange(0,len(archivos_texto)):
        print(i)
        #se extrae el wav 
        audio,fs = librosa.load(archivos_audio[i])
        txt=np.loadtxt(archivos_texto[i])
        #se escogen las frecuencias de interes del wav
        senal_frec=Carga_Filt(audio,fs)
        #se hace el filtrado de wavelet
        senal_filt=Filtrado_wavelet(senal_frec)
#        nombre='archivo'+str(i)+'.wav'#para crear los archivos wav y escuchar el filtrado
#        librosa.output.write_wav(nombre, senal_filt,fs)
        #de esa senal se sacan todos los ciclos
        ciclos=Informacion_sujeto(txt,senal_filt,fs)
        #a cada ciclo se le va a sacar los indices
        for ciclo in ciclos.keys():
            estertor_actual   = ciclos[ciclo][1]#extraccion de estertor
            silibancia_actual = ciclos[ciclo][2]#extraccion de silibancia
            if estertor_actual==0 and silibancia_actual==0:
                estado=0;
            if estertor_actual==0 and silibancia_actual==1:
                estado=1;
            if estertor_actual==1 and silibancia_actual==0:
                estado=2;
            if estertor_actual==1 and silibancia_actual==1:
                estado=3;
                
            varianza,rango,sma,promedio_espectral=Indices(ciclos[ciclo][0],fs)
            VARIANZAS   = np.append(VARIANZAS,varianza)
            RANGOS      = np.append(RANGOS,rango)
            SMAS        = np.append(SMAS,sma)
            PROMEDIOS   = np.append(PROMEDIOS,promedio_espectral)
            
            ESTADOS     = np.append(ESTADOS,estado)
#            ESTERTORES  = np.append(ESTERTORES,estertor_actual)
#            SILIBANCIAS = np.append(SILIBANCIAS,silibancia_actual)
            
        #_______________fin del archivo________________________
        VARIANZAS   = np.append(VARIANZAS,'')
        RANGOS      = np.append(RANGOS,'')
        SMAS        = np.append(SMAS,'')
        PROMEDIOS   = np.append(PROMEDIOS,'')
        
        ESTADOS     = np.append(ESTADOS,'')
#        SILIBANCIAS = np.append(SILIBANCIAS,'')
#        ESTERTORES  = np.append(ESTERTORES,'')
#        if i==4:
#            break
        
    DATA={'VARIANZAS':VARIANZAS,'RANGOS':RANGOS,'SMAS':SMAS,'PROMEDIOS':PROMEDIOS,'ESTADO':ESTADOS}
    return(DATA)
        
ruta_archivos = r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\respiratory-sound-database\Respiratory-Sound-Database\Respiratory-Sound-database\audio-and-txt-files'
DATA=Funcion_madre(ruta_archivos)
df = pd.DataFrame(DATA)  
df.to_csv('DATA', header=True, index=True, sep='\t', mode='a')
#______________________________________________________________________________
#archivo_trabajo=pd.read_csv(r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\DATA3.csv',delimiter=';')





