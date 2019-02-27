import cv2

# import SciPy

I = cv2.imread('mandril.jpg')
cv2.imshow("Mandril", I)
cv2.waitKey(0)

cv2.imwrite("m.png", I)
# debbuger - wyswietlanie skladowych obrazka I[:,:,0] lub I[:,:,1] itp

print(I.shape)  # rozmiary /wiersze, kolumny, glebia/
print(I.size)  # liczba bajtow
print(I.dtype)  # typ danych

IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

# konwersja barw
cv2.imshow('szare', IG)
cv2.waitKey(0)
cv2.imshow('hsv', IHSV)
cv2.waitKey(0)

# zmiana wielkosci
height, width = I.shape[:2]
scale = 1.75
Ix2 = cv2.resize(I, (int(scale * height), int(scale * width)))
cv2.imshow("BigMandril", Ix2)
cv2.waitKey(0)

# I_2 = scipy.misc.imresize(I, 0.5) #z biblioteki SciPy
