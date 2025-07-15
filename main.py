#imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
from scipy.fft import fft,fftfreq
from scipy.signal import butter,filtfilt,sosfilt


def load_file(filename):
    file = pn.read_csv(filename, header=None, names=["t", "x", "y", "z"])
    return(
        file["t"].values[:10000],
        file["x"].values[:10000],
        file["y"].values[:10000],
        file["z"].values[:10000]
    )

def fft_transform(x):
    N = len(x)
    fp = 1.0/20000
    yf = fft(x)
    xf = fftfreq(N,fp)[:N//2]
    yf2 = 2.0 / N * np.abs(yf[0:N//2])
    return xf,yf2


'''
    Filtr do sygnału
        data        = sygnal do filtracji
        fs          = czest. probkowania
        freqLimit   = czest. odciecia
        filterType  = typ filtra // low lub lp - dolnoprzepustowy, high lub hp - gornoprzepustowy
        order       = rzad filtra
'''
def signal_filter(data,fs,freqLimit, filterType, order):
    # sos = butter(order, freqLimit ,btype=filterType,fs = fs, output='sos')
    # filtered = sosfilt(sos,data)
    # return filtered

    Wn = freqLimit/ (fs/2)
    b,a = butter(order, Wn, btype=filterType, analog=False)
    return filtfilt(b,a,data)


# Main program
t,x,y,z = load_file("res/normalne11.csv")
t2,x2,y2,z2 = load_file("res/lozysko_2.csv")

plt.figure("Normalnie")
plt.subplot(3,1,1)
plt.plot(t,x)
plt.subplot(3,1,2)
plt.plot(t,y)
plt.subplot(3,1,3)
plt.plot(t,z)

# plt.figure("Uszkodzone")
# plt.subplot(3,1,1)
# plt.plot(t2,x2)
# plt.subplot(3,1,2)
# plt.plot(t2,y2)
# plt.subplot(3,1,3)
# plt.plot(t2,z2)
#
xFFT,txFFT = fft_transform(x)
yFFT,tyFFT = fft_transform(y)
zFFT,tzFFT = fft_transform(z)

plt.figure("FFT")
plt.subplot(3,1,1)
plt.plot(xFFT,txFFT)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,2)
plt.plot(yFFT,tyFFT)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,3)
plt.plot(zFFT,tzFFT)
plt.grid()
plt.xlim(0,2000)

xFiltr = signal_filter(y, 20000, 500,'lp',10)
yFiltr = signal_filter(y, 20000, 500,'lp',10)
zFiltr = signal_filter(z, 20000, 500,'lp',10)

xFiltrFFT,txFiltrFFT = fft_transform(xFiltr)
yFiltrFFT,tyFiltrFFT = fft_transform(yFiltr)
zFiltrFFT,tzFiltrFFT = fft_transform(zFiltr)

plt.figure("FFT Filtr")
plt.subplot(3,1,1)
plt.plot(xFiltrFFT,txFiltrFFT)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,2)
plt.plot(yFiltrFFT,tyFiltrFFT)
plt.grid()
plt.xlim(0,2000)

plt.subplot(3,1,3)
plt.plot(zFiltrFFT,tzFiltrFFT)
plt.grid()
plt.xlim(0,2000)

plt.show()


print(t)
print(x)
print(y)
print(z)

