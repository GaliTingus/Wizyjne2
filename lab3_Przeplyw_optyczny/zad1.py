import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

W2 = 1
dX = dY = 1

I = cv.imread('I.jpg')
I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
J = cv.imread('J.jpg')
J = cv.cvtColor(J, cv.COLOR_BGR2GRAY)
XX, YY = I.shape

cv.imshow("roznica", cv.absdiff(I, J))

u = np.zeros((XX, YY))
v = np.zeros((XX, YY))

for j in range(W2 * 2, XX - (W2 * 2)):
    for i in range(W2 * 2, YY - (W2 * 2)):
        IO = np.float32(I[j - W2:j + W2 + 1, i - W2:i + W2 + 1])
        cnt = 0
        for x in range(j - dX, j + dX + 1):
            for y in range(i - dY, i + dY + 1):
                JO = np.float32(J[x - W2:x + W2 + 1, y - W2:y + W2 + 1])
                if cnt == 0:
                    distance = np.sum(np.sqrt((np.square(JO - IO))))
                    distance_old = distance
                else:
                    distance = np.sum(np.sqrt((np.square(JO - IO))))

                if distance <= distance_old:
                    v[j, i] = x - j
                    u[j, i] = y - i
                    distance_old = distance

                cnt += 1

plt.quiver(u, v)
plt.gca().invert_yaxis()
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
