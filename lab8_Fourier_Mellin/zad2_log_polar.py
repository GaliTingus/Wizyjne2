import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h,h))

def highpassFilter(size):
    rows = np.cos(np.pi*np.matrix([-0.5 + x/(size[0]-1) for x in range(size[0])]))
    cols = np.cos(np.pi*np.matrix([-0.5 + x/(size[1]-1) for x in range(size[1])]))
    X = np.outer(rows,cols)
    return (1.0 - X) * (2.0 - X)

wz = cv.imread('obrazy_Mellin/domek_r0_64.pgm')
wz = cv.cvtColor(wz, cv.COLOR_BGR2GRAY)

dom = cv.imread('obrazy_Mellin/domek_r30.pgm')
dom = cv.cvtColor(dom, cv.COLOR_BGR2GRAY)

tmp = wz*hanning2D(wz.shape[0])
wz_0 = np.zeros(dom.shape)
wz_0[0:wz.shape[0], 0:wz.shape[1]] = tmp

filtr_wz = highpassFilter(wz_0.shape)
filtr_dom =highpassFilter(dom.shape)

fft_wz = np.fft.fft2(wz_0)
fft_dom = np.fft.fft2(dom)

fft_wz_filter = fft_wz*filtr_wz
fft_dom_filter = fft_dom*filtr_dom

# TO DO
srodek = (obraz_abs_z_fft.shape[0]//2, obraz_abs_z_fft.shape[1]//2)
M = obraz_abs_z_fft.shape[0]/np.log(obraz_abs_z_fft.shape[0]//2)
cv.logPolar(obraz_abs_z_fft, srodek, M, cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)

plt.figure()
plt.imshow(wz, 'gray')
plt.figure()
plt.imshow(wz_0, 'gray')
plt.show()