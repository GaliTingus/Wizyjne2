import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


# def normalize_contours(contours):
#     x = contours[:, 0, 0]
#     y = contours[:, 0, 1]
#
#     dis_x = np.max(x)
#     dis_y = np.max(y)
#
#     M = cv.moments(contours)
#     cx = int(M['m10'] / M['m00'])
#     cy = int(M['m01'] / M['m00'])
#
#     x = (x - cx) / dis_x
#     y = (y - cy) / dis_y
#
#     cx = cx   # zostawić w oryginale, do sprawdzenia
#     cy = cy   # zostawić w oryginale, do sprawdzenia
#     return [x, y, cx, cy]
def normalize_contours(contours):
    x = contours[:, 0, 0]
    y = contours[:, 0, 1]
    tmp = []
    for i in range(len(x)):
        for ii in range(len(x)):
            tmp.append(np.power(x[ii] - x[i], 2) + np.power(y[ii] - y[i], 2))
    distance = np.sqrt(np.max(tmp))

    # print("dist",distance)
    M = cv.moments(contours)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # print(np.max(x/distance), np.max(y/distance))
    x = (x - cx) / distance
    y = (y - cy) / distance

    cx = cx  # zostawić w oryginale, do sprawdzenia
    cy = cy  # zostawić w oryginale, do sprawdzenia
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


# kryternia przetwania obliczen (blad+liczba iteracji)

img_oryginal = cv.imread('Aegeansea.jpg')
# konwersja do odcieni szarosci
img = cv.cvtColor(img_oryginal, cv.COLOR_BGR2HSV)

# ret, thresh1 = cv.threshold(img[:, :, 1], 30, 255, cv.THRESH_BINARY)
# ret, thresh2 = cv.threshold(img[:, :, 0], 60, 255, cv.THRESH_BINARY)
# BIN = np.uint8(cv.multiply(thresh1 / 255, ~thresh2 / 255) * 255)
lower = np.array([0, 25, 0])
upper = np.array([85, 255, 255])

BIN = cv.inRange(img, lower, upper)
kernel = np.ones((3, 3), np.uint8)
BIN = cv.morphologyEx(BIN, cv.MORPH_CLOSE, kernel)
BIN = cv.morphologyEx(BIN, cv.MORPH_OPEN, kernel)
# cv.imshow("1", thresh1)
# cv.imshow("2", thresh2)
# cv.imshow("BIN", BIN)


contours, hierarchy = cv.findContours(BIN, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
contours = list(filter(lambda el: el.shape[0] > 15 and el.shape[0] < 3000, contours))  # filtracja po wielkości
contours = sorted(contours, key=lambda el: el.shape[0], reverse=False)
# cv.drawContours(img_oryginal, contours, -1, (0, 0, 255), 1)
# cv.imshow("Kontury", img_oryginal)
plt.imshow(cv.cvtColor(img_oryginal, cv.COLOR_BGR2RGB))
plt.show()
distance = float('Inf')



image_list = os.listdir('imgs')
for img in range(12):#range(len(image_list)):
    print("szukanie wyspy {} - {}...".format(img,image_list[img].split('.')[0].split('_')[1]))
    img_wz = cv.imread('imgs/' + image_list[img])
    img_wz = cv.cvtColor(img_wz, cv.COLOR_BGR2GRAY)
    contours_wz, hierarchy_wz = cv.findContours(~img_wz, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    X_wz, Y_wz, CX_wz, CY_wz = normalize_contours(contours_wz[0])  # OBRAZ WZORCOWY
    WZOR = np.array([X_wz, Y_wz])
    distance = float('Inf')
    for i in range(len(contours)-3):
        x, y, cx, cy = normalize_contours(contours[i])
        dist = compute_hausdorff(WZOR, np.array([x, y]))
        # print("skanowanie obiektu {} z {} \t dystans = {}".format(i, len(contours), dist))
        if dist < distance:
            distance = dist
            gx = cx
            gy = cy
            index = i
        if distance < 0.090:
            print("Wczesne trafienie")
            break
    if distance == float('Inf'):
        print("Nie znaleziono")
    else:
        if distance<0.090:
            cv.putText(img_oryginal, image_list[img].split('.')[0].split('_')[1], (int(gx), int(gy)),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv.drawContours(img_oryginal, contours, index, (0, 255, 0), 2)
        else:
            cv.putText(img_oryginal, image_list[img].split('.')[0].split('_')[1], (int(gx), int(gy)),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv.drawContours(img_oryginal, contours, index, (0, 0, 255), 2)

cv.imwrite('result.png', img_oryginal)
plt.imshow(cv.cvtColor(img_oryginal, cv.COLOR_BGR2RGB))
plt.show()
