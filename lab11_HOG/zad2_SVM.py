
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
import scipy.ndimage.filters as filters
import math
import scipy

# per00060.ppm
img = cv2.imread('./pedestrians/pos/per00060.ppm')

# plt.figure()
# plt.imshow(img, 'gray')


def HOGpicture(w, bs):  # w - histogramy gradientow obrazu, bs - rozmiar komorki (u nas 8)
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


def single_grad(img):
    dx = scipy.ndimage.filters.convolve1d(np.int32(img), np.array([-1, 0, 1]), 1)
    dy = scipy.ndimage.filters.convolve1d(np.int32(img), np.array([-1, 0, 1]), 0)
    # macierz gradientu
    grad = np.sqrt(dx ** 2 + dy ** 2)
    grad[0:1, 0:-1] = 0
    grad[img.shape[0] - 1 - 1, 0:-1] = 0
    grad[0:-1, 0:1] = 0
    grad[0:-1, img.shape[1] - 1 - 1] = 0
    # orient = np.degrees(np.arctan2(dy, dx))
    orient = np.rad2deg(np.arctan2(dy, dx))
    # orient=orient%180

    # plt.figure()
    # plt.imshow(grad, 'gray')
    # plt.title('GRAD')
    # plt.figure()
    # plt.imshow(orient, 'gray')
    # plt.title('ORIENT')

    return grad, orient


def gradient(img):
    gradB, orientB = single_grad(img[:, :, 0])
    gradG, orientG = single_grad(img[:, :, 1])
    gradR, orientR = single_grad(img[:, :, 2])

    grad = gradB.copy()  # poczatkowo wynikowa macierz to gradientskladowej B
    m1 = gradB - gradG  # m1 - tablica pomocnicza do wyznaczania maksimummiedzy skladowymi B i G
    grad[m1 < 0] = gradG[
        m1 < 0]  # w macierzy wynikowej gradienty skladowej B sapodmieniane na wieksze od nich gradienty skladowej G
    m2 = grad - gradR
    grad[m2 < 0] = gradR[m2 < 0]

    orient = orientB.copy()  # poczatkowo wynikowa macierz to gradientskladowej B
    orient[m1 < 0] = orientG[
        m1 < 0]  # w macierzy wynikowej gradienty skladowej B sapodmieniane na wieksze od nich gradienty skladowej G
    orient[m2 < 0] = orientR[m2 < 0]

    # plt.figure()
    # plt.imshow(grad, 'gray')
    # plt.title('GRAD')
    # plt.figure()
    # plt.imshow(orient, 'gray')
    # plt.title('ORIENT')

    return grad, orient


def HOGhist(img):
    SXY, DIR = gradient(img)
    YY, XX, zz = img.shape

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
            M = SXY[jj * cellSize:(jj + 1) * cellSize, ii * cellSize:(ii + 1) * cellSize]
            T = DIR[jj * cellSize:(jj + 1) * cellSize, ii * cellSize:(ii + 1) * cellSize]
            M = M.flatten()
            T = T.flatten()

            # Obliczenie histogramu
            for k in range(0, cellSize * cellSize):
                # print('k: ',k)
                # print('jj: ',jj)
                # print('ii: ',ii)
                # print('M: ',M)
                m = M[k];
                t = T[k];

                # Usuniecie ujemnych kata (zalozenie katy w stopniach)
                if (t < 0):
                    t = t + 180

                # Wyliczenie przezdialu
                t0 = np.floor((t - 10) / 20) * 20 + 10;  # Przedzial ma rozmiar 20,srodek to 20

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

    return F


gradf, orientf = gradient(img)

F = HOGhist(img)

#######################################################

HOG_data = np.zeros([2 * 100, 3781], np.float32)
for i in range(0, 100):
    IP = cv2.imread('./pedestrians/pos/per%05d.ppm' % (i + 1))
    IN = cv2.imread('./pedestrians/neg/neg%05d.png' % (i + 1))
    F = HOGhist(IP)
    HOG_data[i, 0] = 1
    HOG_data[i, 1:] = F
    F = HOGhist(IN)
    HOG_data[i + 100, 0] = 0
    HOG_data[i + 100, 1:] = F

labels = HOG_data[:, 0]
data = HOG_data[:, 1:]

clf_all_data = svm.SVC(kernel='linear', C=1.0)
clf = svm.SVC(kernel='linear', C=1.0)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

clf_all_data.fit(data, labels)
y_pred = clf_all_data.predict(X_test)

TP = ((y_pred.astype(np.bool) & y_test.astype(np.bool)) == True).sum()
P = (y_pred.astype(np.bool) == True).sum()
TN = ((y_pred.astype(np.bool) & y_test.astype(np.bool)) == False).sum()
N = (y_pred.astype(np.bool) == False).sum()
FP = ((np.logical_not(y_pred.astype(np.bool)) & y_test.astype(np.bool)) == True).sum()
FN = ((y_pred.astype(np.bool) & np.logical_not(y_test.astype(np.bool))) == True).sum()
ACC = (TN+TP)/(P+N)*100

print('Learning using the entire data:')
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("ACC: ", ACC, '%')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

TP = ((y_pred.astype(np.bool) & y_test.astype(np.bool)) == True).sum()
P = (y_pred.astype(np.bool) == True).sum()
TN = ((y_pred.astype(np.bool) & y_test.astype(np.bool)) == False).sum()
N = (y_pred.astype(np.bool) == False).sum()
FP = ((np.logical_not(y_pred.astype(np.bool)) & y_test.astype(np.bool)) == True).sum()
FN = ((y_pred.astype(np.bool) & np.logical_not(y_test.astype(np.bool))) == True).sum()
ACC = (TN+TP)/(P+N)*100

print('Learning using the train data:')
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("ACC: ", ACC, '%')

step = 8

img = cv2.imread('testImage1.png')
# cv2.rectangle(img, (100, 100), (164, 228), (255, 0, 0), 2)
# plt.figure()
# plt.imshow(img)

img_resize = cv2.resize(img, (int(img.shape[1] / 1.3), int(img.shape[0] / 1.3)))
# plt.figure()
# plt.imshow(img_resize)
img_result = img_resize.copy()
for x in range(0, img_resize.shape[1] - 64, step):
    for y in range(0, img_resize.shape[0] - 128, step):
        img_seg = img_resize[y:y + 128, x:x + 64]
        hog = HOGhist(img_seg)
        # y_pred = clf.predict(hog)
        y_pred = clf.predict(hog.reshape(1, hog.shape[0]))
        if y_pred:
            print('y_pred: ', y_pred)
            print('x,y: ', x, y)
            cv2.rectangle(img_result, (x, y), (x + 64, y + 128), (255, 0, 0), 2)
plt.figure()
plt.imshow(img_result)
plt.show()
plt.savefig('output.png')