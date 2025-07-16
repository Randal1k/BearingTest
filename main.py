#imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
from scipy.fft import fft,fftfreq
from scipy.signal import butter,filtfilt
from scipy.stats import skew, kurtosis
from tqdm import tqdm

def load_file(filename):
    try:
        file = pn.read_csv(filename, header=None, names=["t", "x", "y", "z"])

        return file.sort_values('t')
    except Exception as e:
        print(f'Error Loading {filename}: {e}')
        return None

def get_folder_name(folderName):
    folder = folderName.lower()
    if folder == 'normal':
        return 'normal'
    elif folder == 'bearing':
        return 'bearing'
    elif folder == 'unbalance':
        return 'unbalance'
    elif folder == 'misalignment':
        return 'misalignment'
    else:
        return 'normal'

def load_data(folderPath):
    allFiles = []

    failedFile = []
    featureList = []
    labels = []

    for subfolder in os.listdir(folderPath):
        subfolderPath = os.path.join(folderPath,subfolder)

        if os.path.isdir(subfolderPath):
            csvFiles = [f for f in os.listdir(subfolderPath) if f.endswith('.csv')]

            for filename in csvFiles:
                filePath = os.path.join(subfolderPath, filename)
                allFiles.append((filePath, subfolder))

    #Process

    for filePath, folderName in tqdm(allFiles, desc="Processing files"):
        try:
            # Load signal data
            signal = load_file(filePath)
            if signal is None:
                failedFile.append(filePath)
                continue

            # Extract features
            features = process_signal(signal)
            featureList.append(features)

            # Get label from folder name
            label = get_folder_name(folderName)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {filePath}: {e}")
            failedFile.append(filePath)

    return featureList, labels

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

def process_signal(data):
    allFeatures = {}
    for axis in ['x', 'y', 'z']:
        time_signal = data[axis]

        fftFreq, fftMag = signal_FFT(time_signal)
        normalizedSignal = signal_normalization(fftMag)

        feature = calculate_features(normalizedSignal, axis)
        allFeatures.update(feature)

    return allFeatures

def plot_axis_features(axis, featureList,labels):
    axis_features = {}
    # plotsNames = ['mean', 'std', 'rms', 'p2p', 'if', 'skewness', 'Kurtosis', 'crest_factor', 'shape_factor']
    # featN= np.arange(0, 27, 3)
    if featureList:
        firstSample = featureList[0]
        featureNames = [key for key in firstSample.keys() if key.endswith(f'_{axis}')]

        # Extract values for each X-axis feature
        for Name in featureNames:
            axis_features[Name] = []

            for sample in featureList:
                axis_features[Name].append(sample.get(Name, 0))

    fig, axs = plt.subplots(3,3, figsize=(15,8))
    axs = axs.flatten()
    fig.set_size_inches(8, 6)

    startBear = labels.count('bearing')
    startNorm = startBear + labels.count('normal')
    startUnbal = startNorm + labels.count('unbalance')
    startMisal =  startUnbal + labels.count('misalignment')
    print(startBear)
    print(startNorm)
    print(startUnbal)
    print(startMisal)


    for i, (plotname, values) in enumerate(axis_features.items()):
        bearing = values[:startBear]
        normal = values[startBear:startNorm]
        unbal = values[startNorm:startUnbal]
        misal = values[startUnbal:]

        axs[i].plot(bearing, 'g' )
        axs[i].plot(normal, 'b' )
        axs[i].plot(unbal, 'r' )
        axs[i].plot(misal, 'orange')
        axs[i].set_title(plotname)
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Value')
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()
    return axis_features

# Main program
# signal = load_file('res/normalne11.csv')
#
# xFFT_freq,xFFT_mag = signal_FFT(signal.x)
# yFFT_freq,yFFT_mag = signal_FFT(signal.y)
# zFFT_freq,zFFT_mag = signal_FFT(signal.z)
#
#
# xFiltr = signal_filter(signal.x, 20000, 500,'lp',10)
# yFiltr = signal_filter(signal.y, 20000, 500,'lp',10)
# zFiltr = signal_filter(signal.z, 20000, 500,'lp',10)
#
#
# # FFT po filtracji
# xFiltrFFT,txFiltrFFT = signal_FFT(xFiltr)
# yFiltrFFT,tyFiltrFFT = signal_FFT(yFiltr)
# zFiltrFFT,tzFiltrFFT = signal_FFT(zFiltr)
#
# # Normalizacja
# xNorm = signal_normalization(xFFT_mag)
# yNorm = signal_normalization(yFFT_mag)
# zNorm = signal_normalization(zFFT_mag)

features, labels = load_data('res/data/')



#print(xFeatures)
# print("Features =================================")
print(features)
# print("Labels =================================")
print(labels)
xFeatures = plot_axis_features('x', features, labels)
print("")

