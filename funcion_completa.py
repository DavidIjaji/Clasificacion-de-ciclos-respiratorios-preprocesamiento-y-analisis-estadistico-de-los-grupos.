# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:00:30 2020

@author: pc
"""
#Se importan librerias a usar
import librosa;
import librosa.display;
import numpy as np;
from linearFIR import filter_design, mfreqz;
import scipy.signal as signal;
import pywt;
import glob
import pandas as pd
import matplotlib.pyplot as plt;
import scipy.stats as stats;
import pywt;
import glob;
import pandas as pd;
import seaborn as sns;


def Carga_Filt(y,sr): 
    """
    Funcion que permite filtrar la señal de audio y tomar las frecuencias de interes
    Ingresa una señal de audio y su frecuencia de muestreo.
    """
    fs = sr;   # Frecuencia de muestreo
    # Diseño de filtros pasa bajas y altas para tomar la frecuencia de interes
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 1000, revfilt = 0);
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);
    
    # Se aplica la funcion de filtrado filtfilt a los 2 diseños de filtro para 
     #evitar desfases en la señal
    y_hp = signal.filtfilt(highpass,1,y);
    y_bp = signal.filtfilt(lowpass,1,y_hp);
    
    y_bp = np.asfortranarray(y_bp); # Convierte la salida del filtro en una matriz en orden Fortran
    return (y_bp) # salida de la funcion


def Filtrado_wavelet(data):
    """
    Funcion que permite filtrar la señal de acuerdo a su energia para eliminar 
    el ECG, Recibe una matriz filtrada o sin filtrar.
    """
    LL = int(np.floor(np.log2(data.shape[0]))); # Toma las filas argumento ingresado se extrae el logaritmo en base 2 y a
                               # este resultado se lo aproxima al entero mas cercano por debajo de este y se convierte a entero
                               # para obtener el nivel de descomposicion
                                    

    coeff = pywt.wavedec(data, 'db6', level=LL);   # Se hace la transformación de datos multinivel 1D Wavelet, se ingresa la 
                                                    # entrada de la funcion, el wavelet a usar y el nivel de descomposicion
    
    thr = thselect(coeff);   # Se aplica la funcion thselect ingresando la salida de la transformación wavelet
    coeff_t = wthresh(coeff,thr);   # Se aplica la funcion wthresh ingresando la salida de la funcion de la linea anterior
    
    x_rec = pywt.waverec( coeff_t, 'db6'); # Se hace la transformación de datos multinivel 1D Wavelet, se ingresa la salida de
                                    # la linea anterior y el wavelet a usar, el nivel de descomposicion es cero
    
    x_rec = x_rec[0:data.shape[0]];#Se toma los datos desde el inicio hasta la medida de la primer columna de la salida anterior

    x_filt = np.squeeze(data - x_rec); #Se resta la linea anterior a los datos de entrada y se elimina entradas unidimensionales

    return(x_filt)   # Salida de la funcion
    
def wthresh(coeff,thr):
    """
    Funcion que toma la salida de la funcion thselect para para multiplicar sus
    elementos y sumarlos a una nueva lista
    """
    y   = list();   # Se inicializa una lista vacia
    s = wnoisest(coeff);   # Se aplica la funcion wnoisest
    for i in range(0,len(coeff)):    # Ciclo for que recorre un vector de longitud del argumento de entrada de la funcion 
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));   # Se agrega a la lista y la multiplicación de cada 
                                        #posición del arreglo de entrada por su misma posicion en magnitud mientras este
                                        #es recorrido por el ciclo for
    return y;   # Salida de la funcion
    
def thselect(signal):
    """
    Funcion que en lista su argumento de entrada
    """
    Num_samples = 0;   # Se inicializa la variable
    for i in range(0,len(signal)):   # Ciclo for que recorre un vector de longitud del argumento de entrada de la funcion 
        Num_samples = Num_samples + signal[i].shape[0];  # Suma a la variable inicializada cada valor de la primer columna de la 
                                                         #señal de entrada
    
    thr = np.sqrt(2*(np.log(Num_samples)))   # Se encuentra la raiz cuadrada de la salida del ciclo for que se obtiene el 
                                            #logaritmo en base diez y se multiplica por 2
    return thr   # Salida de la funcion

def wnoisest(coeff):
    """
    Funcion que obtiene la media del valor absoludo de su argumento de entrada
    """
    stdc = np.zeros((len(coeff),1));  #Hace una matriz de ceros de una columna y filas como la longitud del argumento de entrada
    for i in range(1,len(coeff)):   # Ciclo for que recorre un vector de longitud del argumento de entrada de la funcion 
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;   # Recorre el argumento de entradao obteniendo su magnitud 
                                                            # y encontrando su media para dividirla sobre 0.6745 y agregarla a la
                                                            #matriz de ceros creada                                                
    return stdc;   # Salida de la funcion

def Informacion_sujeto(texto,audio,fs):
    """
    Funcion que saca los ciclos tomando la información de los archivos de texto, los audios y la 
    frecuencia de muestreo para agregarlos a un diccionario
    """
    Ts=1/fs   # Periodo de muestreo
    info={}   # Se inicializa un diccionario vacio
    i=1   # Contador
    for linea in texto:   # Ciclo for que recorre los archivos de texto
        info['ciclo'+str(i)]=[audio[int(linea[0]/Ts):int(linea[1]/Ts)],int(linea[2]),int(linea[3])]   # Agrega al diccionario 
                                            #los ciclos de acuerdo al archivo de texto, y los datos si posee crepetancia o
                                            #o sibilancia desde cada señal de auido
        i=i+1   # Aumenta en uno el contador
    return(info)   # Salida de la funcion
    
def Indices(ciclo,fs):
    """
    Funcion que obtiene la varianza, rango, un vector y el promedio espectral
    Ingresa un ciclo de la señal y la frecuencia de muestreo
    """
    varianza = np.round(np.var(ciclo),10)   # Varianza con 10 decimales del primer argumento de entrada
    rango = np.round((np.absolute(np.max(ciclo)-np.min(ciclo))),5)   # Rango del ciclo obtenido por el valor absoludo entre su
                                                                      #maximo y minimo redondeado a 5 decimales
    tam_ventana=100   # Tamaño de ventana
    SMA=np.round(max(promedio_movil(ciclo,tam_ventana)),6)   # Usa la funcion promedio_movil para encontrar el maximo de los promedios moviles 
    f, Pxx_spec = signal.welch(ciclo, fs, nperseg=1024, scaling='spectrum')   # Aplica el metodo welch al cilco de entrada con 
                                                                              #la frecuencia de muestreo de entrada, de longitud 
                                                                              #1024 y escala por defecto
    promedio_espectral=np.round((sum(Pxx_spec)/len(Pxx_spec)),11)   # Promedio espectras obtenido por la sumatoria de la densidad
                                                                    #espectral del metodo welch sobre la longitud de este y 
                                                                    #aproximado a 11 decimales
    return(varianza,rango,SMA,promedio_espectral)   # Salida de la funcion
    

def promedio_movil(ciclo, tamano):
    """
    Función que recorre una lista con una ventana de unos con tamaño definido,
    que entrega un generador de valores que son los promedios de cada ventana
    """
    for m in range(len(ciclo)-(tamano-1)):
        yield sum(np.ones(tamano) * ciclo[m:m+tamano]) / tamano # entrega un generador que entrega el promedio de cada ventana
    
def Funcion_madre(ruta_archivos):
    """
    Funcion que aplica las funciones anteriores y filtra cada archivo de audio
    """
    #se generan listas de todos los archivos txt y wav de la carpeta
    archivos_texto = glob.glob(ruta_archivos + '\*.txt')
    archivos_audio = glob.glob(ruta_archivos + '\*.wav') 
    VARIANZAS   = np.array([])
    RANGOS      = np.array([])
    SMAS        = np.array([])
    PROMEDIOS   = np.array([])
    ESTADOS     = np.array([])
#    ESTERTORES = np.array([])
#    SIBILANCIAS= np.array([])
    for i in np.arange(0,len(archivos_texto)):   # Ciclo for que va recorriendo archivo por archivo
        print(i) #Imprime el valor del contador para ver el avance
        #se extrae el wav 
        audio,fs = librosa.load(archivos_audio[i])   # Abre y carga los archivos de audio
        txt=np.loadtxt(archivos_texto[i])
        #se escogen las frecuencias de interes del wav
        senal_frec=Carga_Filt(audio,fs)
        #se hace el filtrado de wavelet
        senal_filt=Filtrado_wavelet(senal_frec)

        
        ciclos=Informacion_sujeto(txt,senal_filt,fs)   # Se sacan todos los ciclos
        #a cada ciclo se le va a sacar los indices
        for ciclo in ciclos.keys():
            estertor_actual   = ciclos[ciclo][1]   # Extraccion de estertor
            sibilancia_actual = ciclos[ciclo][2]   # Extraccion de sibilancia
            if estertor_actual==0 and sibilancia_actual==0:
                estado=0;
            if estertor_actual==0 and sibilancia_actual==1:
                estado=1;
            if estertor_actual==1 and sibilancia_actual==0:
                estado=2;
            if estertor_actual==1 and sibilancia_actual==1:
                estado=3;
                
            varianza,rango,sma,promedio_espectral=Indices(ciclos[ciclo][0],fs)   # Aplicación de la funcion indices
            VARIANZAS   = np.append(VARIANZAS,varianza)
            RANGOS      = np.append(RANGOS,rango)
            SMAS        = np.append(SMAS,sma)
            PROMEDIOS   = np.append(PROMEDIOS,promedio_espectral)
            
            ESTADOS     = np.append(ESTADOS,estado)
#            ESTERTORES  = np.append(ESTERTORES,estertor_actual)
#            SIBILANCIAS = np.append(SIBILANCIAS,silibancia_actual)
            
        #_______________fin del archivo________________________
        #   Se agrega un espacio entre valores añadidos
        VARIANZAS   = np.append(VARIANZAS,'')
        RANGOS      = np.append(RANGOS,'')
        SMAS        = np.append(SMAS,'')
        PROMEDIOS   = np.append(PROMEDIOS,'')
        
        ESTADOS     = np.append(ESTADOS,'')
#        SILIBANCIAS = np.append(SILIBANCIAS,'')
#        ESTERTORES  = np.append(ESTERTORES,'')
        
        
    DATA={'VARIANZAS':VARIANZAS,'RANGOS':RANGOS,'SMAS':SMAS,'PROMEDIOS':PROMEDIOS,'ESTADO':ESTADOS}  # Creacion de un diccionario
                                                                                                     #para los datos obtenidos
    return(DATA)   # Salida de la funcion
        
    # Se aplican las funciones
#ruta_archivos = r'C:\U\Septimo_Nivel\Bioseñales_y_Sistemas\Proyecto_Final\Respiratory_Sound_Database\audio_and_txt_files'
#                        # ruta de la carpeta donde se encuentra la base de datos
#DATA=Funcion_madre(ruta_archivos)   # Se llama a la funcion Funcion_madre
#df = pd.DataFrame(DATA)   # Se añade el diccionario de salida de la funcion aplicada a un dataframe
#df.to_csv('DATA', header=True, index=True, sep='\t', mode='a')   # Se convierte a un archivo .CSV
##______________________________________________________________________________
#archivo_trabajo=pd.read_csv(r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\DATA3.csv',delimiter=';')

archivo_trabajo=pd.read_csv(r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\DATA.csv',delimiter='\t')
archivo_trabajo.head()#se ve el data frrame
sns.heatmap(archivo_trabajo.isnull(), cbar=False)
plt.show()#los NaN son filas que se pusieron para diferenciar entre archivos asi que se eliminan
archivo_trabajo = archivo_trabajo.dropna()
sns.heatmap(archivo_trabajo.isnull(), cbar=False)
plt.show()
archivo_trabajo.drop(['Unnamed: 0'], axis=1, inplace=True)
archivo_trabajo.head()

##ESTADISTICA DESCRIPTIVA
informacion=archivo_trabajo.describe()#muestra la informacion de cuartiles
estados=archivo_trabajo['ESTADO'].value_counts()
#Guardar informacion
informacion.to_csv(r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\cuartiles.csv', header=True, index=True, sep='\t', mode='a')

#Ver si el algortmo funciona igual despues de varios intentos
dat1=pd.read_csv(r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\DATA.csv',delimiter='\t')
dat2=pd.read_csv(r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\DATA3.csv',delimiter=';')
dat1 = dat1.dropna()
dat2 = dat2.dropna()
dat1.drop(['Unnamed: 0','ESTADO'], axis=1, inplace=True)
dat2.drop(['Unnamed: 0','SILIBANCIAS','ESTERTORES'], axis=1, inplace=True)
print(dat1.describe())
print(dat2.describe())
#Los resultados del algoritmo son invarinates 


#Diagrama de caja de cada variable respecto al promedio espectral
sns.boxplot(x='ESTADO',y='PROMEDIOS',data=archivo_trabajo)
plt.plot()
#Diagrama de caja de cada variable respecto a la varianza
sns.boxplot(x='ESTADO',y='VARIANZAS',data=archivo_trabajo)
plt.plot()
    
#Historgramas
# absoluto
i=1
titulos=['VARIANZAS','RANGOS','SMAs','PROM. ESPECTRAL']
for variable in archivo_trabajo.columns[0:4]:
    count,bin_edges = np.histogram(archivo_trabajo[str(variable)])
    plt.figure(1)
    plt.subplot(2,2,i)
    archivo_trabajo[variable].plot(kind='hist')
    plt.xlabel(titulos[i-1])
    plt.ylabel('Cantidad')
    plt.grid()
    i=i+1
    plt.subplots_adjust(left=0, bottom=0, right=1, top=2, wspace=1, hspace=None)


 #Histograma relativo al promedio de cada caracteristica
prom_espec_relativo = archivo_trabajo['PROMEDIOS']/(sum(archivo_trabajo['PROMEDIOS'])/len(archivo_trabajo['PROMEDIOS']))
varianza_relativa = archivo_trabajo['VARIANZAS']/(sum(archivo_trabajo['VARIANZAS'])/len(archivo_trabajo['VARIANZAS']))
rango_relativa = archivo_trabajo['RANGOS']/(sum(archivo_trabajo['RANGOS'])/len(archivo_trabajo['RANGOS']))
SMA_relativa = archivo_trabajo['SMAS']/(sum(archivo_trabajo['SMAS'])/len(archivo_trabajo['SMAS']))
#Dataframe de informacion relativa
data_relative={'VARIANZA':varianza_relativa,'RANGO':rango_relativa,'SMA':SMA_relativa,'PROMEDIOS ESPECTRALES':prom_espec_relativo}
relativa = pd.DataFrame(data_relative)  


caracteristicas=[prom_espec_relativo,varianza_relativa,rango_relativa,SMA_relativa]
i=1
for variable in caracteristicas:
    count,bin_edges = np.histogram(variable)
    plt.figure(1)
    plt.subplot(2,2,i)
    variable.plot(kind='hist')
    plt.xlabel(titulos[i-1])
    plt.ylabel('Cantidad')
    plt.grid()
    i=i+1
    plt.subplots_adjust(left=0, bottom=0, right=1, top=2, wspace=1, hspace=None)
   
    
#Agrupar por estado
df_mean = archivo_trabajo[['VARIANZAS','RANGOS','SMAS','PROMEDIOS','ESTADO']]
info_estado = df_mean.groupby(['ESTADO'],as_index=False).mean()
print(info_estado)
info_estado.to_csv(r'C:\Users\pc\OneDrive - Universidad de Antioquia\7mo semestre\Bioseñales\TrabajoFinal\info_estado.csv', header=True, index=True, sep='\t', mode='a')

#Correlaicon
correlation_matrix = archivo_trabajo.corr()
ax=sns.heatmap(correlation_matrix, annot=True)
ax.set_ylim(0, 5)
ax.set_xlim(0, 5)
plt.show()

#grafico de dispercion relativo
plt.scatter(prom_espec_relativo,varianza_relativa,color='r')
plt.ylabel('PROMEDIO ESPECTRAL RELATIVO')
plt.xlabel('VARIANZA RELATIVA')
plt.title('DISPERSION DE VARIANZA Y PROM. ESPECTRAL RELATIVOS')
plt.show()

plt.scatter(rango_relativa,varianza_relativa,color='r')
plt.ylabel('VARIANZA RELATIVO')
plt.xlabel('RANGO RELATIVO')
plt.title('DISPERSION DE VARIANZA Y RANGO RELATIVOS')
plt.show()

#Test de spearman 
i=0
for variable in archivo_trabajo.columns:
    for variable2 in archivo_trabajo.columns[i:]:
        if not variable == variable2: 
            cor, pval = stats.spearmanr(archivo_trabajo[variable2],archivo_trabajo[variable])
            print('Evaluando: '+str(variable)+' vs '+str(variable2))
            print('Test Spearman: cor:'+str(np.round(cor,2))+'\t pval: '+str(np.round(pval,2)))
    i=i+1

from statsmodels.graphics.gofplots import qqplot
qqplot(archivo_trabajo['VARIANZAS'], line='s')

#discriminacion de la informacion 
info_sano=archivo_trabajo.loc[archivo_trabajo.loc[:,'ESTADO']==0]
info_silibancias=archivo_trabajo.loc[archivo_trabajo.loc[:,'ESTADO']==1]
info_crepitancias=archivo_trabajo.loc[archivo_trabajo.loc[:,'ESTADO']==2]
info_crep_sili=archivo_trabajo.loc[archivo_trabajo.loc[:,'ESTADO']==3]

#TEST DE MANNWHITNEYU

#SANO VS SILIBANCIAS

for variable in archivo_trabajo.columns:
    statistic , pvalues = stats.mannwhitneyu(info_sano[variable],info_silibancias[variable])
    print(variable+' vs '+variable)
    print('MANNWHITENEYU: '+'stat: '+str(statistic)+'  pval: '+str(pvalues))
    

#SANO VS CREPITANCIAS

for variable in archivo_trabajo.columns:
    statistic , pvalues = stats.mannwhitneyu(info_sano[variable],info_crepitancias[variable])
    print(variable+' vs '+variable)
    print('MANNWHITENEYU: '+'stat: '+str(statistic)+'  pval: '+str(pvalues))

#SILIBANCIAS VS CREPITANCIAS

for variable in archivo_trabajo.columns:
    statistic , pvalues = stats.mannwhitneyu(info_crepitancias[variable],info_silibancias[variable])
    print(variable+' vs '+variable)
    print('MANNWHITENEYU: '+'stat: '+str(statistic)+'  pval: '+str(pvalues))

#PRUEBA EN UN MISMO GRUPO PARA REVIZAR LA CONFIABILIDAD DE LOS RESULTADOS
statistic , pvalues = stats.mannwhitneyu(info_silibancias['PROMEDIOS'][0:443],info_silibancias['PROMEDIOS'][443:])
print(variable+' vs '+variable)
print('MANNWHITENEYU: '+'stat: '+str(statistic)+'  pval: '+str(pvalues))
#LOS RESULTADOS NO SON CONFIABLES 



