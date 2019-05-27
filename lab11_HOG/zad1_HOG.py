import cv2 as cv
import numpy as np
import scipy.ndimage
import math
import matplotlib.pyplot as plt
import os


def HOGpicture(w, bs):  # w - histogramy gradientow obrazu, bs - rozmiarkomorki (u nas 8)
    bim1 = np.zeros((bs, bs))
    bim1[np.round(bs // 2):np.round(bs // 2) + 1, :] = 1;
    bim = np.zeros(bim1.shape + (9,));
    bim[:, :, 0] = bim1;
    for i in range(0, 9):  # 2:9,
        bim[:, :, i] = scipy.misc.imrotate(bim1, -i * 20, 'nearest') / 255
        Y, X, Z = w.shape
        w[w < 0] = 0;
        im = np.zeros((bs * Y, bs * X));
    for i in range(Y):
        iisl = (i) * bs
        iisu = (i + 1) * bs
        for j in range(X):
            jjsl = j * bs
            jjsu = (j + 1) * bs
            for k in range(9):
                im[iisl:iisu, jjsl:jjsu] += bim[:, :, k] * w[i, j, k];
    return im




def hog(im):
    XX = im.shape[1]
    YY = im.shape[0]

    # im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    dx = scipy.ndimage.filters.convolve1d(np.int32(im), np.array([-1, 0, 1]), 1)
    dy = scipy.ndimage.filters.convolve1d(np.int32(im), np.array([-1, 0, 1]), 0)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    orient = np.rad2deg(np.arctan2(dy, dx))

    orientR = orient[:, :, 0]
    orientG = orient[:, :, 1]
    orientB = orient[:, :, 2]
    gradR = grad[:, :, 0]
    gradG = grad[:, :, 1]
    gradB = grad[:, :, 2]

    grad = gradB.copy()  # poczatkowo wynikowa macierz to gradient skladowej B
    m1 = gradB - gradG  # m1 - tablica pomocnicza do wyznaczania maksimum miedzy skladowymi B i G
    grad[m1 < 0] = gradG[
        m1 < 0]  # w macierzy wynikowej gradienty skladowej B sa podmieniane na wieksze od nich gradienty skladowej G
    m2 = grad - gradR
    grad[m2 < 0] = gradR[m2 < 0]

    orient = orientB.copy()
    orient[m1 < 0] = orientG[m1 < 0]
    orient[m2 < 0] = orientR[m2 < 0]

    # Obliczenia histogramow
    cellSize = 8  # rozmiar komorki
    YY_cell = np.int32(YY / cellSize)
    XX_cell = np.int32(XX / cellSize)
    # Kontener na histogramy - zakladamy, ze jest 9 przedzialow
    hist = np.zeros([YY_cell, XX_cell, 9], np.float32)
    # Iteracja po komorkach na obrazie
    for jj in range(0, YY_cell):
        for ii in range(0, XX_cell):
            # Wyciecie komorki
            M = grad[jj * cellSize:(jj + 1) * cellSize, ii * cellSize:(ii + 1) * cellSize]
            T = orient[jj * cellSize:(jj + 1) * cellSize, ii * cellSize:(ii + 1) * cellSize]
            M = M.flatten()
            T = T.flatten()
            # Obliczenie histogramu
            for k in range(0, cellSize * cellSize):
                m = M[k];
                t = T[k];
                # Usuniecie ujemnych kata (zalozenie katy w stopniach)
                if (t < 0):
                    t = t + 180
                # Wyliczenie przezdialu
                t0 = np.floor((t - 10) / 20) * 20 + 10;  # Przedzial ma rozmiar 20, srodek to 20
                # Przypadek szczegolny tj. t0 ujemny
                if (t0 < 0):
                    t0 = 170
                # Wyznaczenie indeksow przedzialu
                i0 = int((t0 - 10) / 20);
                i1 = i0 + 1;
                # Zawijanie
                if i1 == 9:
                    i1 = 0;
                # Obliczenie odleglosci do srodka przedzialu
                d = min(abs(t - t0), 180 - abs(t - t0)) / 20
                # Aktualizacja histogramu
                hist[jj, ii, i0] = hist[jj, ii, i0] + m * (1 - d)
                hist[jj, ii, i1] = hist[jj, ii, i1] + m * (d)


    # Normalizacja w blokach
    e = math.pow(0.00001, 2)
    F = []
    for jj in range(0, YY_cell - 1):
        for ii in range(0, XX_cell - 1):
            H0 = hist[jj, ii, :]
            H1 = hist[jj, ii + 1, :]
            H2 = hist[jj + 1, ii, :]
            H3 = hist[jj + 1, ii + 1, :]
            H = np.concatenate((H0, H1, H2, H3))
            n = np.linalg.norm(H)
            Hn = H / np.sqrt(math.pow(n, 2) + e)
            F = np.concatenate((F, Hn))

    # obraz = HOGpicture(hist, 8)
    # plt.imshow(obraz)
    # plt.show()

    return F



