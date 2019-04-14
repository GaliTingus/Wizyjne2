import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def filtracja(BIN):
    BIN = cv.medianBlur(BIN, 5)
    kernel = np.ones((7, 7), np.uint8)
    BIN = cv.morphologyEx(BIN, cv.MORPH_CLOSE, kernel)
    BIN = cv.morphologyEx(BIN, cv.MORPH_OPEN, kernel)
    # BIN = cv.erode(BIN, kernel, iterations=2)
    # BIN = cv.dilate(BIN, kernel, iterations=2)
    # kernel = np.ones((7, 7), np.uint8)
    # BIN = cv.erode(BIN, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    # BIN = cv.erode(BIN, kernel, iterations=1)
    # BIN = cv.medianBlur(BIN, 7)
    return BIN


def ramki(I, BIN, show_labels=False):
    output = cv.connectedComponentsWithStats(BIN, 4, cv.CV_32S)
    min_thresh = 1800
    max_thresh = 99999999
    # if show_labels: cv.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))
    if (output[0] > 1):  # czy sa jakies obiekty
        for i in range(output[0]):
            if output[2][i][4] >= min_thresh and output[2][i][4] <= max_thresh and output[2][i][3] > output[2][i][2]:
                cv.rectangle(I, (output[2][i][0], output[2][i][1]),
                             (output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (255, 0, 0), 2)
        # tab = stats[1:, 4]  # wyciecie 4 kolumny bez pierwszego elementu
        # pi = np.argmax(tab)  # znalezienie indeksu najwiekszego elementu
        # pi = pi + 1  # inkrementacja bo chcemy indeks w stats, a nie w tab
        # # wyrysownie bbox
        # cv.rectangle(I, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2],
        #                                                stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
        # # wypisanie informacji o polu i numerze najwiekszego elementu
        # cv.putText(I, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        # cv.putText(I, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])),
        #            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    return I


# arr = []
# for x,y,w,h in contourRects:
#       arr.append((x,y))
#       arr.append((x+w,y+h))
#
# box = cv.minAreaRect(np.asarray(arr))
# pts = cv.boxPoints(box) # 4 outer corners


cap = cv.VideoCapture('vid1_IR.avi')
while (cap.isOpened()):
    ret, frame = cap.read()
    G = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    BIN = cv.threshold(G, 45, 255, cv.THRESH_BINARY)
    BIN = BIN[1]
    BIN = filtracja(BIN)
    cv.imshow('IR', G)
    cv.imshow("binaryzacja", BIN)
    cv.imshow("ruch", ramki(G, BIN))
    if cv.waitKey(1) & 0xFF == ord('q'):  # przerwanie petli po wcisnieciu klawisza ’q’
        break
cap.release()
