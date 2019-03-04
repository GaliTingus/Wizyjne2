import cv2 as cv
import numpy as np

#Wczytanie i wyswietlenie sekwencji
# for i in range(550,1700,4):
#     I = cv.imread('input/in%06d.jpg' % i)
#     cv.imshow("I",I)
#     cv.waitKey(10)


#Pierwsza klatka
i = 400
old = cv.imread('input/in%06d.jpg' % i)
old = cv.cvtColor(old, cv.COLOR_BGR2GRAY)

#Pętla po obrazach
for i in range(401,1090):
    I = cv.imread('input/in%06d.jpg' % i)
    I2 = cv.cvtColor(I, cv.COLOR_BGR2GRAY)  #Zamiana na czarnobiałe
    new = cv.absdiff(I2, old)               #Roznica
    BIN = cv.threshold(new,10,255,cv.THRESH_BINARY) #Progowanie
    BIN = BIN[1]

    kernel = np.ones((3, 3), np.uint8)
    BIN = cv.erode(BIN, kernel, iterations=1)
    BIN = cv.dilate(BIN, kernel, iterations=7)
    kernel = np.ones((7, 7), np.uint8)
    BIN = cv.erode(BIN, kernel, iterations=2)
    BIN = cv.medianBlur(BIN, 5)
    cv.imshow("bin", BIN)

    retval, labels, stats, centroids = cv.connectedComponentsWithStats(BIN)
    #cv.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))
    if (stats.shape[0] > 1):   #czy sa jakies obiekty
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
    cv.imshow("WYNIK",I)
    old = I2
    cv.waitKey(12)
