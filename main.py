#imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
from scipy.fft import fft,fftfreq


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

plt.figure("Uszkodzone")
plt.subplot(3,1,1)
plt.plot(t2,x2)
plt.subplot(3,1,2)
plt.plot(t2,y2)
plt.subplot(3,1,3)
plt.plot(t2,z2)

x_fft,t_fft = fft_transform(x)
y_fft,t2_fft = fft_transform(y)
z_fft,t3_fft = fft_transform(z)

plt.figure("XFFT")
plt.plot(x_fft,t_fft)
plt.grid()
plt.xlim(0,2000)

plt.figure("YFFT")
plt.plot(y_fft,t2_fft)
plt.grid()
plt.xlim(0,2000)


plt.show()


print(t)
print(x)
print(y)
print(z)

