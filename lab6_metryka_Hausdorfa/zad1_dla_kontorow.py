import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# kryternia przetwania obliczen (blad+liczba iteracji)

img_oryginal = cv.imread('ithaca_q.bmp')
# konwersja do odcieni szarosci
img = cv.cvtColor(img_oryginal, cv.COLOR_BGR2GRAY)
# cv.imshow("obraz", img)

im2, contours, hierarchy = cv.findContours(~img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(img_oryginal, contours, 0, (0, 0, 255), 1)


# cv.imshow("kontury", img_oryginal)

def normalize_contours(contours):
    x = contours[0][:, 0, 0]
    y = contours[0][:, 0, 1]
    # tmp = []
    # for i in range(len(x)):
    #     for ii in range(len(x)):
    #         tmp.append(np.power(x[ii] - x[i], 2) + np.power(y[ii] - y[i], 2))
    # distance = np.sqrt(np.max(tmp))
    dis_x=np.max(x)
    dis_y=np.max(y)
    # print("dist",distance)
    M = cv.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # print(np.max(x/distance), np.max(y/distance))
    x = (x - cx) / dis_x
    y = (y - cy) / dis_y

    cx = cx / dis_x  # zostawić w oryginale, do sprawdzenia
    cy = cy / dis_y  # zostawić w oryginale, do sprawdzenia
    # print("X={} Y={} cx={} cy={}".format(x, y, cx, cy))
    return [x, y, cx, cy]


def compute_hausdorff(Obj1, Obj2):
    h = 0.0
    for i in range(Obj1.shape[1]):
        shortest = float('Inf')
        for j in range(Obj2.shape[1]):
            d = np.sqrt(np.power(Obj2[0][j] - Obj1[0][i], 2) + np.power(Obj2[1][j] - Obj1[1][i], 2))
            if d < shortest:
                shortest = d
        if shortest > h:
            h = shortest
    return h


X, Y, CX, CY = normalize_contours(contours)     #OBRAZ WZORCOWY
ITHACA = np.array([X, Y])
distance_list = []
image_list = os.listdir('imgs')
for i in range(len(image_list)):
    tmp_img = cv.imread('imgs/' + image_list[i])
    tmp_img = cv.cvtColor(tmp_img, cv.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv.findContours(~tmp_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    x, y, cx, cy = normalize_contours(contours)
    # tmp = np.array([x,y])
    distance_list.append(compute_hausdorff(ITHACA, np.array([x, y])))

index = np.argmin(distance_list)
print("Nalepiej dopasowany obraz: {} o metryce = {}".format(image_list[index], distance_list[index]))
