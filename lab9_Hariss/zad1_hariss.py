import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage.filters as filters


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
    plt.plot(xy[1],xy[0],'*',color='r')
    return 0


img_oryg_1 = cv.imread('budynek1.jpg')
img_1 = cv.cvtColor(img_oryg_1, cv.COLOR_BGR2GRAY)

img_oryg_2 = cv.imread('budynek2.jpg')
img_2 = cv.cvtColor(img_oryg_2, cv.COLOR_BGR2GRAY)

img_new = Haris(img_1, ksize=5, k=0.05)
img_new=img_new/np.max(img_new)
max_1=find_max(img_new, 9, .1)
# print(max_1)
# plt.imshow(img_new,'gray')
print_dot(img_oryg_1,max_1)

img_new_2 = Haris(img_2, ksize=5, k=0.05)
img_new_2=img_new_2/np.max(img_new_2)
max_2=find_max(img_new_2, 9, .1)
# print(max_2)
# plt.imshow(img_new_2,'gray')
print_dot(img_oryg_2,max_2)
plt.show()
