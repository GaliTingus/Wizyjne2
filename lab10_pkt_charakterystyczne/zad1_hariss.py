import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage.filters as filters
import pm


def Haris(img, ksize, k):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=ksize)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=ksize)
    Ixx = gx * gx
    Ixx = cv.GaussianBlur(Ixx, (ksize, ksize), 0)
    Iyy = gy * gy
    Iyy = cv.GaussianBlur(Iyy, (ksize, ksize), 0)
    Ixy = gx * gy
    Ixy = cv.GaussianBlur(Ixy, (ksize, ksize), 0)

    det = Ixx * Iyy - (Ixy ** 2)
    trace = Ixx + Iyy

    H = det - k * trace * trace
    return H


def find_max(image, size, threshold):  # size - rozmiar maski filtramaksymalnego
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def print_dot(img, xy):
    plt.figure()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.plot(xy[1], xy[0], '*', color='r')
    return 0


def get_description(img, XY, size):
    X = img.shape[1]
    Y = img.shape[0]
    pts = XY
    l_otoczen = []
    l_wspolrzednych = []
    pts = list(filter(lambda pt: pt[0] >= size and pt[0] < Y - size and pt[1] >= size and pt[1] < X - size,
                      zip(pts[0], pts[1])))
    for i in range(len(pts)):
        l_otoczen.append((img[pts[i][0] - size:pts[i][0] + size, pts[i][1] - size:pts[i][1] + size]).flatten())
        l_wspolrzednych.append(pts[i])

    wynik_funkcji = list(zip(l_otoczen, l_wspolrzednych))
    return wynik_funkcji


def compare_description(list_1, list_2, n):
    l_wynikow = []
    for i in range(len(list_1)):
        for j in range(len(list_2)):
            tmp = list_2[j][0]-list_1[i][0]
            wynik = np.sum(np.power(tmp,2))
            l_wynikow.append([wynik,[i,j]])

    l_wynikow.sort(key=lambda x: x[0])
    return l_wynikow[0:n]

img_oryg_1 = cv.imread('eiffel1.jpg')
img_1 = cv.cvtColor(img_oryg_1, cv.COLOR_BGR2GRAY)

img_oryg_2 = cv.imread('eiffel2.jpg')
img_2 = cv.cvtColor(img_oryg_2, cv.COLOR_BGR2GRAY)

img_new = Haris(img_1, ksize=5, k=0.05)
img_new = img_new / np.max(img_new)
max_1 = find_max(img_new, 9, .1)
# print(max_1)
# plt.imshow(img_new,'gray')
print_dot(img_oryg_1, max_1)

img_new_2 = Haris(img_2, ksize=5, k=0.05)
img_new_2 = img_new_2 / np.max(img_new_2)
max_2 = find_max(img_new_2, 9, .1)
# print(max_2)
# plt.imshow(img_new_2,'gray')
print_dot(img_oryg_2, max_2)

list_1 = get_description(img_oryg_1, max_1, 15)
list_2 = get_description(img_oryg_2, max_2, 15)
tmp = compare_description(list_1,list_2,20)

lista = []
n=20
for i in range(n):
    xy1 = list_1[tmp[i][1][0]][1]
    xy2 = list_2[tmp[i][1][1]][1]
    lista.append([[xy1[0],xy1[1]],[xy2[0],xy2[1]]])

pm.plot_matches(img_oryg_1,img_oryg_2,lista)
plt.show()
