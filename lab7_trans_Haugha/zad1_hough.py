import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os



img_oryginal = cv.imread('trybik.jpg')
# konwersja do odcieni szarosci
img = cv.cvtColor(img_oryginal, cv.COLOR_BGR2GRAY)
# cv.imshow("obraz", img)
BIN = cv.threshold(img, 240, 255, cv.THRESH_BINARY)
BIN=BIN[1]
BIN = cv.medianBlur(BIN, 3)


sobelx = cv.Sobel(BIN,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(BIN,cv.CV_64F,0,1,ksize=5)

grad = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
grad = grad/np.abs(np.max(grad))
orint = np.arctan2(sobely,sobelx)

im2, contours, hierarchy = cv.findContours(BIN, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(img_oryginal, contours, 1, (0, 0, 255), 1)
cv.imshow("BIN",BIN)
M = cv.moments(contours[1])
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
cv.imshow("kontury",img_oryginal)

Rtable = [[] for i in range(360)]


contour = contours[1][:, :, :]
for i in range(len(contours[1])):
    d =np.sqrt(np.power(contour[i][0][0]-cx,2)+np.power(contour[i][0][1]-cy,2))
    kat =np.arctan2(contour[i][0][0]-cx, contour[i][0][1]-cy) * 180 / np.pi
    Rtable[index].append([d,kat])
print(kat)

print(np.arctan2(1,1)* 180 / np.pi)





cv.waitKey(0)
cv.destroyAllWindows()