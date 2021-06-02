
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np
import math
wav_files=[]
file =[]
written=0
for r, d, f in os.walk('dataset1/'):
    for i in f:
        wav_files.append(i)
        file.append(i)
    for i in file:
        samprate,data = wavfile.read(os.path.join(r,i))
        if (len(np.shape(data))==2):
            data_size = np.shape(data)[0]
            data = np.ravel(data)
            data=data[:data_size]
        written=written+1
        wavfile.write(i,samprate,data)

            
wav_files=[]
freq_and_pow = {}
for r, d, f in os.walk('my_set/'):
    for i in f:
        wav_files.append(i)
for i in wav_files:
    samprate,data = wavfile.read(os.path.join(r,i))
    freq,Pow = signal.welch(data,samprate)
    freq_and_pow[i]={}
    freq_and_pow[i]['Sample rate']=samprate
    freq_and_pow[i]['frequency_size']=np.size(freq)
    freq_and_pow[i]['Power_size']=np.size(Pow)
    freq_and_pow[i]['Data_size']=np.shape(data)
    
output = open('dataset2.csv', 'w')
output.write("Название файла,")
output.write('Частота дискретизации,')
output.write('Формат по частоте,')
output.write('Формат по мощности')
output.write('Формат данных')
for key in freq_and_pow.keys():
    output.write('\n'+key+',')
    for column in freq_and_pow[key].keys():
        output.write(str(freq_and_pow[key][column])+',')
output.close()

wav_files=[]
power = []
freqs = []
for r, d, f in os.walk('my_set/'):
    for i in f:
        wav_files.append(os.path.join(i))
    for i in wav_files:
        samprate,data = wavfile.read(os.path.join(r,i))
        freq,Pow = signal.welch(data,samprate)
        Pow = Pow/np.sum(Pow)
        power.append(Pow)
        freqs.append(freq)
        
power=np.array(power)
KL_divergence = []
KL_val = 0
for i in range(67):
    for j in range(67):
        for k in range(len(freq)):
            KL_val+=power[i,k]*math.log2(power[i,k]/power[j,k])
        KL_divergence.append(KL_val)
        KL_val = 0

IS_divergence = []
IS_val = 0
for i in range(67):
    for j in range(67):
        for k in range(len(freq)):
            IS_val+=power[i,k]/power[j,k]-math.log2(power[i,k]/power[j,k])-1
        IS_divergence.append(IS_val)
        IS_val =0
    
output = open('results_KL.csv', 'w')
output.write(",")
for i in file:
    output.write(i+',')
for i in range(67):
    output.write('\n'+file[i])
    for j in range(67):
        output.write(','+str(KL_divergence[i*67+j]))
        


output.close()
    
output = open('results_IS.csv', 'w')
output.write(",")
for i in file:
    output.write(i+',')
for i in range(67):
    output.write('\n'+file[i])
    for j in range(67):
        output.write(','+str(IS_divergence[i*67+j]))
        


output.close()

KL_np = np.array(KL_divergence)
KL_np=np.reshape(KL_np,[67,67])
KL_np_t = np.transpose(KL_np)
plt.figure()
plt.scatter(KL_np, KL_np_t,s=0.5)
plt.show()

IS_np = np.array(IS_divergence)
IS_np=np.reshape(IS_np,[67,67])
IS_np_t = np.transpose(IS_np)
plt.figure()
plt.scatter(IS_np, IS_np_t,s=0.5)
plt.show()
