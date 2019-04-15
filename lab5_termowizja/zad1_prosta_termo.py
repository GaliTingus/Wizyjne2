import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

roi_mask = np.vstack((np.zeros((100, 480), dtype=np.uint8), np.ones((360 - 100, 480), dtype=np.uint8)))


def filtracja(BIN):
    BIN = BIN * roi_mask
    BIN = cv.medianBlur(BIN, 5)
    kernel = np.ones((7, 7), np.uint8)
    BIN = cv.morphologyEx(BIN, cv.MORPH_CLOSE, kernel)
    BIN = cv.morphologyEx(BIN, cv.MORPH_OPEN, kernel)
    return BIN


def ramki(I, BIN, show_labels=False):
    output = cv.connectedComponentsWithStats(BIN, 8, cv.CV_32S)
    stats = output[2]
    centroids = output[3]
    for i in range(stats.shape[0]):
        for j in range(stats.shape[0]):
            parent_blob_left, parent_blob_top = stats[i, 0], stats[i, 1]
            parent_blob_right, parent_blob_bottom = parent_blob_left + stats[i, 2], parent_blob_top + stats[i, 3]
            parent_blob_area = stats[i, 4]
            parent_x, parent_y = centroids[i, 0], centroids[i, 1]

            left, top = stats[j, 0], stats[j, 1]
            right, bottom = left + stats[j, 2], top + stats[j, 3]
            area = stats[j, 4]
            x, y = centroids[j, 0], centroids[j, 1]
            if i != j and parent_x - 40 < x < parent_x + 40:
                new_left = parent_blob_left if parent_blob_left < left else left
                new_top = parent_blob_top if parent_blob_top < top else top
                new_right = parent_blob_right if parent_blob_right > right else right
                new_bottom = parent_blob_bottom if parent_blob_bottom > bottom else bottom
                stats[i, :] = np.array(
                    [new_left, new_top, new_right - new_left, new_bottom - new_top, parent_blob_area + area])

    if output[0] > 1:  # czy sa jakies obiekty
        for i in range(output[0]):
            if stats[i][4] >= 400:
                cv.rectangle(I, (stats[i][0], stats[i][1]),
                             (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (255, 0, 0), 2)
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
    BIN = cv.threshold(G, 38, 255, cv.THRESH_BINARY)
    BIN = BIN[1]
    BIN = filtracja(BIN)
    cv.imshow('IR', G)
    cv.imshow("binaryzacja", BIN)
    cv.imshow("ruch", ramki(G, BIN))
    if cv.waitKey(5) & 0xFF == ord('q'):  # przerwanie petli po wcisnieciu klawisza ’q’
        break
cap.release()
