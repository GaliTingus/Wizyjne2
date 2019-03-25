import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

W2 = 1
dX = dY = 1
scale = 5


def pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv.resize(images[k - 1], (0, 0), fx=0.5, fy=0.5))
    return images


def of(I, J, u0, v0, W2=1, dY=1, dX=1):
    u = np.zeros(I.shape, dtype=np.int8)
    v = np.zeros(I.shape, dtype=np.int8)

    for j in range(W2 * 2, I.shape[0] - (W2 * 2)):
        for i in range(W2 * 2, I.shape[1] - (W2 * 2)):
            IO = np.float32(I[j - W2:j + W2 + 1, i - W2:i + W2 + 1])
            cnt = 0
            for x in range(j - dX, j + dX + 1):
                for y in range(i - dY, i + dY + 1):
                    x0 = x + u0[j, i]
                    y0 = y + v0[j, i]
                    # print("xo = ", x0, " y0 = ",y0)
                    JO = np.float32(J[x0 - W2:x0 + W2 + 1, y0 - W2:y0 + W2 + 1])
                    if cnt == 0:
                        distance = np.sum(np.sqrt((np.square(JO - IO))))
                        distance_old = distance
                    else:
                        distance = np.sum(np.sqrt((np.square(JO - IO))))

                    if distance <= distance_old:
                        wsp_y = x - j
                        wsp_x = y - i
                        distance_old = distance

                    cnt += 1
            u[j, i] = wsp_x + u0[j, i]
            v[j, i] = wsp_y + v0[j, i]
    return u, v


I = cv.imread('I.jpg')
I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
J = cv.imread('J.jpg')
J = cv.cvtColor(J, cv.COLOR_BGR2GRAY)
XX, YY = I.shape

I_pyramid = pyramid(I, scale)
J_pyramid = pyramid(J, scale)

u0 = np.zeros(I_pyramid[-1].shape, np.int8)
v0 = np.zeros(J_pyramid[-1].shape, np.int8)

u = None
v = None
u_pyramid = []
v_pyramid = []
start = time.process_time()
for i in reversed(range(scale)):
    u, v = of(I_pyramid[i], J_pyramid[i], u0, v0, W2, dY, dX)
    u0 = cv.resize(u, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
    v0 = cv.resize(v, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)

    u_pyramid.append(u)
    v_pyramid.append(v)

stop = time.process_time()
print("czas = {:.2f} sec.".format(stop - start))

for i in range(scale):
    plt.figure()
    plt.quiver(u_pyramid[i], v_pyramid[i])
    plt.gca().invert_yaxis()
    plt.title("skala = 1/" + str(2 ** (scale - 1 - i)))
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
