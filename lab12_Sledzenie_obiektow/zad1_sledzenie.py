import cv2 as cv
import numpy as np
import scipy.ndimage
# import scipy
import math
import matplotlib.pyplot as plt
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
import sys

kernel_size = 45
mouseX, mouseY = (830, 430)


# rozmiar rozkladu
# przykladowe wspolrzedne
def track_init(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.rectangle(param, (x - kernel_size // 2, y - kernel_size // 2),
                     (x + kernel_size // 2, y + kernel_size // 2), (0, 255, 0), 2)
        mouseX, mouseY = x, y


# Wczytanie pierwszego obrazka
I = cv.imread('seq/track00100.png')
cv.namedWindow('Tracking')
cv.setMouseCallback('Tracking', track_init, param=I)
# Pobranie klawisza
while (1):
    cv.imshow('Tracking', I)
    k = cv.waitKey(20) & 0xFF
    if k == 27:  # ESC
        break

kernel_size = 35  # rozmiar rozkladu
sigma = kernel_size / 6  # odchylenie std
x = np.arange(0, kernel_size, 1, float)  # wektor poziomy
y = x[:, np.newaxis]  # wektor pionowy
x0 = y0 = kernel_size // 2  # wsp. srodka
G = 1 / (2 * math.pi * sigma ** 2) * np.exp(-0.5 * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

xS = mouseX - kernel_size // 2
yS = mouseY - kernel_size // 2
I_HSV = cv.cvtColor(I, cv.COLOR_BGR2HSV)
I_H = I_HSV[:, :, 0]
hist_q = np.zeros((256, 1), float)
for u in range(256):
    mask = I_H[yS:yS + kernel_size, xS:xS + kernel_size] == u
    hist_q[u] = np.sum(G[mask])
hist_q = hist_q / hist_q.max()

for i in range(101, 201):
    I = cv.imread('seq/track%05d.png' % i)
    I_HSV = cv.cvtColor(I, cv.COLOR_BGR2HSV)
    I_H = I_HSV[:, :, 0]

    hist_p = np.zeros((256, 1), float)
    for u in range(256):
        mask = I_H[yS:yS + kernel_size, xS:xS + kernel_size] == u
        hist_p[u] = np.sum(G[mask])
    hist_q = hist_p / hist_p.max()

    shift = np.sqrt(hist_p * hist_q)
    window = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    sum_wind = 0
    sum_x = 0
    sum_y = 0
    for row in range(kernel_size):
        for col in range(kernel_size):
            window[row, col] = G[row, col] * shift[I_H[row, col]]
            sum_wind += window[row, col]
            sum_x += window[row, col] * row
            sum_y += window[row, col] * col

    # xx = int(sum_x / sum_wind)
    # yy = int(sum_y / sum_wind)
    moments = cv.moments(window)
    if moments['m00'] < 0.0000001:
        continue
    xc = int(moments['m10'] / moments['m00'])
    yc = int(moments['m01'] / moments['m00'])
    # print(xc, xx)

    xS = xS + (yc - kernel_size // 2)
    yS = yS + (xc - kernel_size // 2)
    # xS = xS + (yy - kernel_size // 2)
    # yS = yS + (xx - kernel_size // 2)

    cv.rectangle(I, (xS, yS), (xS + kernel_size, yS + kernel_size), (255, 0, 0), 2)

    cv.imshow('Sledzenie', I)
    k = cv.waitKey(1)

    if k == 27:  # ESC
        break
