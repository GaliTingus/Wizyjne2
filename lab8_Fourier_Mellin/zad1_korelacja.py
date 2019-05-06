import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

wz = cv.imread('obrazy_Mellin/wzor.pgm')
wz = cv.cvtColor(wz, cv.COLOR_BGR2GRAY)

dom = cv.imread('obrazy_Mellin/domek_r0.pgm')
dom = cv.cvtColor(dom, cv.COLOR_BGR2GRAY)

tmp = np.zeros(dom.shape)
tmp[0:48, 0:48] = wz

fft1 = np.fft.fft2(dom)
fft2 = np.fft.fft2(tmp)

ccor = np.fft.ifft2(fft1*np.conj(fft2))
R = (np.conj(fft2)*fft1)/np.abs((np.conj(fft2)*fft1))
ccor2 = np.fft.ifft2(R)
y, x = np.unravel_index(np.argmax(np.abs(ccor)), ccor.shape)
y2,x2 =np.unravel_index(np.argmax(np.abs(ccor2)), ccor2.shape)
print(x,y)
print(x2,y2)

macierz_translacji = np.float32([[1,0,x2],[0,1,y2]])  # gdzie dx, dy -wektor przesuniecia
obraz_przesuniety = cv.warpAffine(tmp, macierz_translacji,(tmp.shape[1], tmp.shape[0]))


plt.imshow(tmp,'gray')
plt.figure()
plt.imshow(dom,'gray')
plt.figure()
plt.imshow(np.abs(ccor),'gray')
plt.figure()
plt.imshow(obraz_przesuniety,'gray')
plt.show()
