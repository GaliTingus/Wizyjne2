import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def rgb2gray(I):
    return 0.299*I[:,:,0] + 0.587*I[:,:,1] + 0.114*I[:,:,2]


I = plt.imread('mandril.jpg')
fig,ax = plt.subplots(1)
plt.imshow(I)
plt.title('Małpka')
plt.axis('off')


plt.imsave('mandril.png',I)

x = [100, 150, 200, 250]
y = [50, 100, 150, 200]

plt.plot(x,y,'r.',markersize=10) #SPRAWDZIC

rect = Rectangle((50,50),50,100,fill=False, ec='r')
ax.add_patch(rect)
plt.show()

#Konerwsja do szarosci
plt.figure(2)
plt.gray()
plt.imshow(rgb2gray(I))
plt.show()
