import cv2 as cv
import numpy as np

cv.destroyAllWindows()

img = cv.imread('./trybik.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = 255 - img_gray

img_bin = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
img_bin = img_bin[1]

img_med = cv.medianBlur(img_bin, 5)
img_rev = 255 - img_bin

cv.imshow('Image', img)
cv.imshow('Image bin', img_bin)

m = cv.moments(img_rev, True)
x_centr = m['m10'] / m['m00']
y_centr = m['m01'] / m['m00']

# img_cont_ref contours, hierarchy = cv.findContours(image=img_med, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
contours, hierarchy = cv.findContours(image=img_med, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

# img_cont = cv.drawContours(img_cont_ref/2, contours, 0, (255,0,0))
# cv.imshow('Imagine contures', img_cont)

img = cv.drawContours(image=img, contours=contours, contourIdx=0, color=(255, 0, 0))

cv.circle(img=img, center=(np.int(x_centr), np.int(y_centr)), radius=1, color=(0, 255, 255), thickness=5)
cv.imshow('Image', img)

sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=5)

grad = np.sqrt(sobelx ** 2 + sobely ** 2)
grad = grad / np.max(grad)

orient = np.rad2deg(np.arctan2(sobelx, sobely))
orient = orient + 180
orient = np.uint16(orient)

Rtable = [[] for i in range(360)]

for cont in contours[0]:
    wek_kat = orient[cont[0, 1], cont[0, 0]]
    wek_dist = np.sqrt((cont[0, 0] - x_centr) ** 2 + (cont[0, 1] - y_centr) ** 2)
    wek_orient = np.arctan2(cont[0, 1] - y_centr, cont[0, 0] - x_centr)

    if wek_kat == 360:
        wek_kat = 0
    Rtable[wek_kat].append([wek_dist, wek_orient])

img_2 = cv.imread('./trybiki2.jpg')
img_2_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

sobelx_2 = cv.Sobel(img_2_gray, cv.CV_64F, 1, 0, ksize=5)
sobely_2 = cv.Sobel(img_2_gray, cv.CV_64F, 0, 1, ksize=5)

grad_2 = np.sqrt(sobelx_2 ** 2 + sobely_2 ** 2)
grad_2 = grad_2 / np.max(grad_2)

orient_2 = np.rad2deg(np.arctan2(sobelx_2, sobely_2))
orient_2 = orient_2 + 180
orient_2 = np.uint16(orient_2)

result = np.zeros(img_2.shape[:2], dtype=np.uint16)

for x in range(grad_2.shape[0]):

    for y in range(grad_2.shape[1]):

        if grad_2[x, y] > 0.2:
            col = Rtable[orient_2[x, y]]

            for r, fi in col:
                x1 = int(x + r * np.sin(fi))
                y1 = int(y + r * np.cos(fi))
                if 0 <= x1 < grad_2.shape[0] and 0 <= y1 < grad_2.shape[1]:
                    result[x1, y1] += 1

_result = np.where(result == result.max())

for i in range(len(_result[0])):
    cv.circle(img=img_2, center=(np.int(_result[1][i]), np.int(_result[0][i])), radius=2, color=(255, 0, 255),
               thickness=2)

result = np.uint8(result * 255.0 / result.max())

cv.imshow('Punkty przeciecia', result)
cv.imshow('Trybiki wszystkie', img_2)

cv.waitKey(0)












# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
#
#
# img_oryginal = cv.imread('trybik.jpg')
# # konwersja do odcieni szarosci
# img = cv.cvtColor(img_oryginal, cv.COLOR_BGR2GRAY)
# # cv.imshow("obraz", img)
# BIN = cv.threshold(img, 240, 255, cv.THRESH_BINARY)
# BIN=BIN[1]
# BIN = cv.medianBlur(BIN, 3)
#
#
# sobelx = cv.Sobel(BIN,cv.CV_64F,1,0,ksize=5)
# sobely = cv.Sobel(BIN,cv.CV_64F,0,1,ksize=5)
#
# grad = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
# grad = grad/np.abs(np.max(grad))
# orint = np.arctan2(sobely,sobelx)
#
# im2, contours, hierarchy = cv.findContours(BIN, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# cv.drawContours(img_oryginal, contours, 1, (0, 0, 255), 1)
# cv.imshow("BIN",BIN)
# M = cv.moments(contours[1])
# cx = int(M['m10'] / M['m00'])
# cy = int(M['m01'] / M['m00'])
# cv.imshow("kontury",img_oryginal)
#
# Rtable = [[] for i in range(360)]
#
#
# contour = contours[1][:, :, :]
# for i in range(len(contours[1])):
#     d =np.sqrt(np.power(contour[i][0][0]-cx,2)+np.power(contour[i][0][1]-cy,2))
#     kat =np.arctan2(contour[i][0][0]-cx, contour[i][0][1]-cy) * 180 / np.pi
#     Rtable[index].append([d,kat])
# print(kat)
#
# print(np.arctan2(1,1)* 180 / np.pi)
#
#
#
#
#
# cv.waitKey(0)
# cv.destroyAllWindows()