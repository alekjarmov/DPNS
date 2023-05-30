import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
mpl_use('TkAgg')

img = cv2.imread('../images/example.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if you specify it)
#The image should be a grayscale image
#Number of corners you want to find
#Quality level, which is a value between 0-1, which denotes the minimum quality of corner below which everyone is rejected
#Minimum euclidean distance between corners detected

corners = cv2.goodFeaturesToTrack(gray, 500, 0.01, 10)
print(corners)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

plt.imshow(img), plt.show()