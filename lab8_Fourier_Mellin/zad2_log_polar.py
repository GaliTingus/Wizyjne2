import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
import os
from matplotlib.patches import Rectangle

plt.close('all')


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))


def highpassFilter(size):
    rows = np.cos(np.pi * np.matrix([-0.5 + x / (size[0] - 1) for x in range(size[0])]))
    cols = np.cos(np.pi * np.matrix([-0.5 + x / (size[1] - 1) for x in range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X) * (2.0 - X)


def rotate(x, y, xo, yo, theta):  # rotate x,y around xo,yo by theta (deg)

    xr = np.cos(np.deg2rad(theta)) * (x - xo) - np.sin(np.deg2rad(theta)) * (y - yo) + xo
    yr = np.sin(np.deg2rad(theta)) * (x - xo) + np.cos(np.deg2rad(theta)) * (y - yo) + yo
    return int(xr), int(yr)


img = cv.imread('./obrazy_Mellin/domek_r0_64.pgm', cv.IMREAD_GRAYSCALE)
img_org = cv.imread('./obrazy_Mellin/domek_r0_64.pgm', cv.IMREAD_GRAYSCALE)
house_img = cv.imread('./obrazy_Mellin/domek_r120.pgm', cv.IMREAD_GRAYSCALE)
img = img * hanning2D(img.shape[0])

plt.figure()
plt.imshow(img, 'gray')

result = np.zeros(house_img.shape)
result[0:img.shape[0], 0:img.shape[1]] = img
offset = int((house_img.shape[0] - img.shape[0]) // 2.0)
result_c = np.zeros(house_img.shape)
result_c[offset:img.shape[0] + offset, offset:img.shape[1] + offset] = img

fft_result = fft.fft2(result)
fft_result_sh = fft.fftshift(fft_result)
fft_house = fft.fft2(house_img)
fft_house_sh = fft.fftshift(fft_house)

highpass_filter = highpassFilter(result.shape)
fft_result_abs = np.abs(fft_result_sh) * highpass_filter
fft_house_abs = np.abs(fft_house_sh) * highpass_filter

R = fft_result.shape[0] // 2.0
M = 2 * R / np.log(R)
result_logpol = cv.logPolar(src=fft_result_abs, center=(R, R), M=M, flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)

house_logpol = cv.logPolar(src=fft_house_abs, center=(R, R), M=M, flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS)

fft_result_logpol = fft.fft2(result_logpol)
fft_house_logpol = fft.fft2(house_logpol)

R = (np.conj(fft_result_logpol) * fft_house_logpol) / np.abs(np.conj(fft_result_logpol) * fft_house_logpol)
correl_norm = fft.ifft2(R)

wsp_kata, wsp_logr = np.unravel_index(np.argmax(np.abs(correl_norm)), np.abs(correl_norm).shape)
rozmiar_logr = house_logpol.shape[1]
rozmiar_kata = house_logpol.shape[0]

if wsp_logr > rozmiar_logr // 2:  # rozmiar_logr x rozmiar_kata to rozmiar obrazu logpolar
    wykl = rozmiar_logr - wsp_logr  # powiekszenie
else:
    wykl = - wsp_logr  # pomniejszenie

A = (wsp_kata * 360.0) / rozmiar_kata
scale = np.exp(wykl / M)  # gdzie M to parametr funkcji cv.logPolar, a wykl wyliczamy jako:
kat1 = - A  # gdzie A = (wsp_kata * 360.0 ) /rozmiar_kata
kat2 = 180 - A

srodekTrans = ((result.shape[0] / 2), (result.shape[1] / 2))
macierz_translacji_1 = cv.getRotationMatrix2D((srodekTrans[0], srodekTrans[1]), kat1, scale)
macierz_translacji_2 = cv.getRotationMatrix2D((srodekTrans[0], srodekTrans[1]), kat2, scale)

im_rot_scaled_1 = cv.warpAffine(result_c, macierz_translacji_1, result.shape)
im_rot_scaled_2 = cv.warpAffine(result_c, macierz_translacji_2, result.shape)

im_rot_scaled_fft_1 = np.fft.fft2(im_rot_scaled_1)
im_rot_scaled_fft_2 = np.fft.fft2(im_rot_scaled_2)

R_1 = (np.conj(im_rot_scaled_fft_1) * fft_house) / np.abs((np.conj(im_rot_scaled_fft_1) * fft_house))
ccor_norm_1 = np.fft.ifft2(R_1)
R_2 = (np.conj(im_rot_scaled_fft_2) * fft_house) / np.abs((np.conj(im_rot_scaled_fft_2) * fft_house))
ccor_norm_2 = np.fft.ifft2(R_2)
y_norm_1, x_norm_1 = np.unravel_index(np.argmax(np.abs(ccor_norm_1)), np.abs(ccor_norm_1).shape)
y_norm_2, x_norm_2 = np.unravel_index(np.argmax(np.abs(ccor_norm_2)), np.abs(ccor_norm_2).shape)

if np.max(np.abs(ccor_norm_1)) > np.max(np.abs(ccor_norm_2)):
    x_best = x_norm_1
    y_best = y_norm_1
    kat_best = kat1
    result_best = im_rot_scaled_1
    corr_best = ccor_norm_1
    macierz_translacji_best = macierz_translacji_1
else:
    x_best = x_norm_2
    y_best = y_norm_2
    kat_best = kat2
    result_best = im_rot_scaled_2
    corr_best = ccor_norm_2
    macierz_translacji_best = macierz_translacji_2

# print(np.sqrt((x_best - result_best.shape[0] / 2) ** 2 + (y_best - result_best.shape[0] / 2) ** 2))
# print(np.sqrt((result_best.shape[0] / 2) ** 2 + (result_best.shape[1] / 2) ** 2) - 4)

if np.sqrt((x_best - result_best.shape[0] / 2) ** 2 + (y_best - result_best.shape[0] / 2) ** 2) > np.sqrt(
        (result_best.shape[0] / 2) ** 2 + (result_best.shape[1] / 2) ** 2) - 4:
    x_best = 0
    y_best = result_best.shape[1]

print('x best: ', x_best)
print('y best: ', y_best)

plt.figure()
plt.imshow(img_org, 'gray')
plt.scatter(img_org.shape[0] // 2, img_org.shape[0] // 2, s=20, c='g')

plt.figure()
plt.imshow(house_img, 'gray')
currentAxis = plt.gca()
xy = rotate(x_best, y_best - result_best.shape[1], (x_best + result_best.shape[0] / 2),
            (y_best - result_best.shape[1] + result_best.shape[0] / 2), kat_best)

currentAxis.add_patch(
    Rectangle(xy=xy, width=result_best.shape[0], height=result_best.shape[1], angle=kat_best, fill=None, alpha=1,
              color='red'))
plt.scatter(x_best + result_best.shape[0] // 2, y_best - result_best.shape[1] // 2, s=20, c='m')

plt.figure()
plt.imshow(im_rot_scaled_1, 'gray')
plt.scatter(im_rot_scaled_1.shape[0] // 2, im_rot_scaled_1.shape[0] // 2, s=20, c='m')

# plt.figure()
# plt.imshow(result_best, 'gray')
# plt.scatter(result_best.shape[0] // 2, result_best.shape[0] // 2, s=20, c='m')

# plt.figure()
# plt.imshow(np.abs(corr_best))
# plt.scatter(x_best, y_best, s=40, c='m')

plt.show()
