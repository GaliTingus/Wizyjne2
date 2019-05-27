import cv2 as cv
import numpy as np
import zad1_HOG
import scipy.ndimage
import math

HOG_data = np.zeros([2 * 100, 3781], np.float32)
for i in range(0, 100):
    IP = cv.imread('pos/per%05d.ppm' % (i + 1))
    IN = cv.imread('neg/neg%05d.png' % (i + 1))
    F = hog(IP)
    HOG_data[i, 0] = 1
    HOG_data[i, 1:] = F
    F = hog(IN)
    HOG_data[i + 100, 0] = 0
    HOG_data[i + 100, 1:] = F
