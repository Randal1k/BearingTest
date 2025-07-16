#imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
from scipy.fft import fft,fftfreq
from scipy.signal import butter,filtfilt
from scipy.stats import skew, kurtosis
import  statistics
import math

def load_file(filename):
    try:
        file = pn.read_csv(filename, header=None, names=["t", "x", "y", "z"])

        return file.sort_values('t')
    except Exception as e:
        print(f'Error Loading {filename}: {e}')
        return None
def signal_FFT(data):
    windowed_signal = data * np.hanning(len(data))
    N = len(data)
    fp = 1/20000
    yf = fft(windowed_signal)
    xf = fftfreq(N,fp)
    positive_idx = xf >= 0
    frequencies = xf[positive_idx]
    magnitude = np.abs(yf[positive_idx])

    return frequencies,magnitude

'''
    Filtr do sygnału
        data        = sygnal do filtracji
        fs          = czest. probkowania
        freqLimit   = czest. odciecia
        filterType  = typ filtra // low lub lp - dolnoprzepustowy, high lub hp - gornoprzepustowy
        order       = rzad filtra
'''
def signal_filter(data,fs,freqLimit, filterType, order):
    Wn = freqLimit/ (fs/2)
    b,a = butter(order, Wn, btype=filterType, analog=False)
    return filtfilt(b,a,data)

def signal_normalization(data):
    #return (data - np.min(data)) / (np.max(data) - np.min(data))
    #return (data - np.mean(data)) / np.std(data)
    return data / np.max(data)
def signal_average(data):
    return np.mean(data)

def signal_std_deviation(data):
    return np.std(data)

def signal_RMS(data):
    return  np.sqrt(signal_average(data**2))

def signal_P2P(data):
    return np.max(data) - np.min(data)

def signal_IF(data):
    return np.max(data) / signal_average(data)

def signal_skewness(data):
    return skew(data,axis=0,bias=True)

def signal_kurtosis(data):
    return kurtosis(data,axis=0,bias=True)

def signal_crestFactor(data):
    return abs(np.max(data)) / signal_RMS(data)

def signal_shapeFactor(data):
    return 1/signal_average(data)

def calculate_features(data, axis):
    features = {}

    features[f'mean_{axis}'] = signal_average(data)
    features[f'std_{axis}'] = signal_std_deviation(data)
    features[f'rms_{axis}'] = signal_RMS(data)
    features[f'p2p_{axis}'] = signal_P2P(data)
    features[f'if_{axis}'] = signal_IF(data)
    features[f'skewness_{axis}'] = signal_skewness(data)
    features[f'kurtosis_{axis}'] = signal_kurtosis(data)
    features[f'crest_factor_{axis}'] = signal_crestFactor(data)
    features[f'shape_factor_{axis}'] = signal_shapeFactor(data)

    return features


# Main program
signal = load_file('res/normalne11.csv')
#t,x,y,z = load_file("res/normal_000_Ch08_100g_PE_Acceleration.csv")
#t2,x2,y2,z2 = load_file("res/lozysko_2.csv")


# Normalny sygnal
plt.figure("Normalnie")
plt.subplot(3,1,1)
plt.plot(signal.t,signal.x)
plt.subplot(3,1,2)
plt.plot(signal.t,signal.y)
plt.subplot(3,1,3)
plt.plot(signal.t,signal.z)

plt.tight_layout()

# plt.figure("Uszkodzone")
# plt.subplot(3,1,1)
# plt.plot(t2,x2)
# plt.subplot(3,1,2)
# plt.plot(t2,y2)
# plt.subplot(3,1,3)
# plt.plot(t2,z2)
#


xFFT_freq,xFFT_mag = signal_FFT(signal.x)
yFFT_freq,yFFT_mag = signal_FFT(signal.y)
zFFT_freq,zFFT_mag = signal_FFT(signal.z)


#FFT
plt.figure("FFT")
plt.subplot(3,1,1)
plt.plot(xFFT_freq,xFFT_mag)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,2)
plt.plot(yFFT_freq,yFFT_mag)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,3)
plt.plot(zFFT_freq,zFFT_mag)
plt.grid()
plt.xlim(0,2000)
plt.tight_layout()

xFiltr = signal_filter(signal.x, 20000, 500,'lp',10)
yFiltr = signal_filter(signal.y, 20000, 500,'lp',10)
zFiltr = signal_filter(signal.z, 20000, 500,'lp',10)


# FFT po filtracji
xFiltrFFT,txFiltrFFT = signal_FFT(xFiltr)
yFiltrFFT,tyFiltrFFT = signal_FFT(yFiltr)
zFiltrFFT,tzFiltrFFT = signal_FFT(zFiltr)

# plt.figure("FFT Filtr")
# plt.subplot(3,1,1)
# plt.plot(xFiltrFFT,txFiltrFFT)
# plt.grid()
# plt.xlim(0,2000)
#
# plt.subplot(3,1,2)
# plt.plot(yFiltrFFT,tyFiltrFFT)
# plt.grid()
# plt.xlim(0,2000)
#
# plt.subplot(3,1,3)
# plt.plot(zFiltrFFT,tzFiltrFFT)
# plt.grid()
# plt.xlim(0,2000)


# Normalizacja
xNorm = signal_normalization(xFFT_mag)
yNorm = signal_normalization(yFFT_mag)
zNorm = signal_normalization(zFFT_mag)

plt.figure("Normalization")
plt.subplot(3,1,1)
plt.plot(xFFT_freq,xNorm)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,2)
plt.plot(yFFT_freq,yNorm)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,3)
plt.plot(zFFT_freq,zNorm)
plt.grid()
plt.xlim(0,2000)
plt.tight_layout()

plt.show()

allFeatures = {}

for axis in ['x','y','z']:
    time_signal = signal[axis]

    fftFreq, fftMag = signal_FFT(time_signal)
    normalizedSignal = signal_normalization(fftMag)

    feature = calculate_features(fftMag, axis)
    allFeatures.update(feature)


print(allFeatures)
print("")

