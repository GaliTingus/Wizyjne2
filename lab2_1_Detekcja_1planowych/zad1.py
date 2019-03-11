import cv2 as cv
import numpy as np

# Wczytanie i wyswietlenie sekwencji
# for i in range(550,1700,4):
#     I = cv.imread('input/in%06d.jpg' % i)
#     cv.imshow("I",I)
#     cv.waitKey(10)
TP = TN = FN = FP =0

# Pierwsza klatka
i = 300
old = cv.imread('input/in%06d.jpg' % i)
old = cv.cvtColor(old, cv.COLOR_BGR2GRAY)

# Pętla po obrazach
for i in range(301, 1090):
    I = cv.imread('input/in%06d.jpg' % i)
    REF = cv.imread('groundtruth/gt%06d.png' % i)
    REF = cv.cvtColor(REF, cv.COLOR_BGR2GRAY)  # Zamiana na czarnobiałe
    I2 = cv.cvtColor(I, cv.COLOR_BGR2GRAY)  # Zamiana na czarnobiałe
    new = cv.absdiff(I2, old)  # Roznica
    BIN = cv.threshold(new, 13, 255, cv.THRESH_BINARY)  # Progowanie
    BIN = BIN[1]
    BIN = cv.medianBlur(BIN, 7)
    kernel = np.ones((3, 3), np.uint8)
    #BIN = cv.erode(BIN, kernel, iterations=1)
    BIN = cv.dilate(BIN, kernel, iterations=8)
    kernel = np.ones((7, 7), np.uint8)
    BIN = cv.erode(BIN, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    BIN = cv.erode(BIN, kernel, iterations=2)
    BIN = cv.medianBlur(BIN, 7)
    cv.imshow("bin", BIN)

    retval, labels, stats, centroids = cv.connectedComponentsWithStats(BIN)
    cv.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))
    if (stats.shape[0] > 1):  # czy sa jakies obiekty
        tab = stats[1:, 4]  # wyciecie 4 kolumny bez pierwszego elementu
        pi = np.argmax(tab)  # znalezienie indeksu najwiekszego elementu
        pi = pi + 1  # inkrementacja bo chcemy indeks w stats, a nie w tab
        # wyrysownie bbox
        cv.rectangle(I, (stats[pi, 0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2],
                                                       stats[pi, 1] + stats[pi, 3]), (255, 0, 0), 2)
        # wypisanie informacji o polu i numerze najwiekszego elementu
        cv.putText(I, "%f" % stats[pi, 4], (stats[pi, 0], stats[pi, 1]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv.putText(I, "%d" % pi, (np.int(centroids[pi, 0]), np.int(centroids[pi, 1])),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv.imshow("WYNIK", I)

    height, width = BIN.shape[:2]
    cv.imshow("REF", REF)
    TP_M = np.logical_and((BIN == 255),(REF == 255)) # iloczyn logiczny odpowiednich elementow macierzy
    TP_S = np.sum(TP_M) # suma elementow w macierzy
    TP = TP + TP_S

    TN_M = np.logical_and((BIN == 0), (REF == 0))  # iloczyn logiczny odpowiednich elementow macierzy
    TN_S = np.sum(TN_M)  # suma elementow w macierzy
    TN = TN + TN_S

    FP_M = np.logical_and((BIN == 255), (REF == 0))  # iloczyn logiczny odpowiednich elementow macierzy
    FP_S = np.sum(FP_M)  # suma elementow w macierzy
    FP = FP + FP_S

    FN_M = np.logical_and((BIN == 0), (REF == 255))  # iloczyn logiczny odpowiednich elementow macierzy
    FN_S = np.sum(FN_M)  # suma elementow w macierzy
    FN = FN + FN_S

    old = I2
    cv.waitKey(5)

P = FP/(TP+FP)
R = TP/(FP+FN)
F1= 2*P*R/(P+R)
print("F1 = ", F1)