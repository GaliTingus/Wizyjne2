import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist(img):
    h=np.zeros((256,1), np.float32)
    height, width = img.shape[:2]
    for y in range(height):
        for x in range(width):
            h[img[x,y]] += 1
    return h


I = cv2.imread('mandril.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
lena = cv2.imread('lena.png')
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

cv2.imshow("malpka", I)
cv2.imshow("lena", lena)
cv2.imshow("roznica", np.uint8(I-lena))
cv2.imshow("iloczyn", np.uint8(I*lena))
cv2.imshow("kombinacja", np.uint8(0.2*I+0.8*lena))
cv2.imshow("modul roznicy",cv2.absdiff(I,lena))
plt.hist(hist(I))
plt.show()








cv2.waitKey(0)

