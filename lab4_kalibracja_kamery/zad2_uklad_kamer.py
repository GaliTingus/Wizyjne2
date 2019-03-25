import cv2
import numpy as np
import matplotlib.pyplot as plt

# kryternia przetwania obliczen (blad+liczba iteracji)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# przygotowanie punktow 2D w postaci: (0,0,0), (1,0,0), (2,0,0) ....,(6,7,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# tablice do przechowywania punktow obiektow (3D) i punktow na obrazie (2D) dla wszystkich obrazow
objpoints = []  # punkty 3d w przestrzeni (rzeczywsite)

imgpoints_L = []  # punkty 2d w plaszczyznie obrazu.Kalibracja kamery i stereowizja
imgpoints_R = []  # punkty 2d w plaszczyznie obrazu.Kalibracja kamery i stereowizja

for i in range(1, 13):
    # wczytanie obrazu
    img_L = cv2.imread('images_left/left%02d.jpg' % i)
    img_R = cv2.imread('images_right/right%02d.jpg' % i)
    # konwersja do odcieni szarosci
    gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
    # wyszukiwanie naroznikow na planszy
    ret_L, corners_L = cv2.findChessboardCorners(gray_L, (7, 6), None)
    ret_R, corners_R = cv2.findChessboardCorners(gray_R, (7, 6), None)
    # jesli znaleniono na obrazie punkty
    if (ret_L == True & ret_R == True):
        # dolaczenie wspolrzednych 3D
        objpoints.append(objp)
        # poprawa lokalizacji punktow (podpiskelowo)
        corners2_L = cv2.cornerSubPix(gray_L, corners_L, (11, 11), (-1, -1), criteria)
        corners2_R = cv2.cornerSubPix(gray_R, corners_R, (11, 11), (-1, -1), criteria)
        # dolaczenie poprawionych punktow
        imgpoints_L.append(corners2_L)
        imgpoints_R.append(corners2_R)
        # dolaczenie poprawionych punktow
        cv2.drawChessboardCorners(img_L, (7, 6), corners2_L, ret_L)
        cv2.drawChessboardCorners(img_R, (7, 6), corners2_R, ret_R)
        # wizualizacja wykrytych naroznikow
        # cv2.drawChessboardCorners(img_L, (7, 6), corners2_L, ret_L)
        # cv2.imshow("Corners", img_L)
    # if ret_R == True:
    #     # dolaczenie wspolrzednych 3D
    #     objpoints_R.append(objp)
    #     # poprawa lokalizacji punktow (podpiskelowo)
    #
    #     # wizualizacja wykrytych naroznikow
    #     # cv2.drawChessboardCorners(img_L, (7, 6), corners2_L, ret_L)
    #     # cv2.imshow("Corners", img_L)

ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv2.calibrateCamera(objpoints, imgpoints_L, gray_L.shape[::-1], None, None)
ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv2.calibrateCamera(objpoints, imgpoints_R, gray_R.shape[::-1], None, None)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F =cv2.stereoCalibrate(objpoints,imgpoints_L,imgpoints_R,mtx_L,dist_L,mtx_R,dist_R,gray_L.shape[::-1])

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray_R.shape[::-1], R, T)

map1_L, map2_L = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1,P1, gray_L.shape[::-1], cv2.CV_16SC2)
map1_R, map2_R = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2,P2, gray_L.shape[::-1], cv2.CV_16SC2)

i=1
img_L = cv2.imread('images_left/left%02d.jpg' % i)
img_R = cv2.imread('images_right/right%02d.jpg' % i)

dst_L = cv2.remap(img_L, map1_L, map2_L, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_R, map1_R, map2_R, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1] # pobranie rozmiarow obrazka (kolorowego)
visRectify = np.zeros((YY,XX*2,N),np.uint8) # utworzenie nowego obrazka o szerokosci x2
visRectify[:,0:640:,:] = dst_L
# przypisanie obrazka lewego
visRectify[:,640:1280:,:] = dst_R
# przypisanie obrazka prawego
# Wyrysowanie poziomych linii
for y in range(0,480,10):
    cv2.line(visRectify, (0,y), (1280,y), (255,0,0))
    cv2.imshow('visRectify',visRectify)
#wizualizacja


# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print("ret  ={} \nmtx = {} \ndist = {}, \nrvecs = {} \ntvecs = {}".format(ret, mtx, dist, rvecs, tvecs))
#
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
# cv2.imwrite('calibresult.png', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
