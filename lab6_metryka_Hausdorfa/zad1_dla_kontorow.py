import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# kryternia przetwania obliczen (blad+liczba iteracji)

img_oryginal = cv.imread('ithaca_q.bmp')
# konwersja do odcieni szarosci
img = cv.cvtColor(img_oryginal, cv.COLOR_BGR2GRAY)
cv.imshow("obraz", img)

im2, contours, hierarchy = cv.findContours(~img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(img_oryginal, contours, 0, (0, 0, 255), 1)
# cv.imshow("kontury", img_oryginal)

def normalize_contours(contours):
    x = contours[0][:, 0, 0]
    y = contours[0][:, 0, 1]
    tmp=[]
    for i in range(len(x)):
        for ii in range(len(x)):
            tmp.append(np.sqrt(np.power(x[ii] - x[i],2) + np.power(y[ii] - y[i],2)))
    distance = np.max(tmp)
    # print("dist",distance)
    M = cv.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    x = (x - cx) / distance
    y = (y - cy) / distance
    cx = cx / distance  #zostawić w oryginale, do sprawdzenia
    cy = cy / distance  #zostawić w oryginale, do sprawdzenia
    # print("X={} Y={} cx={} cy={}".format(x, y, cx, cy))
    return [x,y,cx,cy]

print(normalize_contours(contours))
plt.imshow(img_oryginal)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
