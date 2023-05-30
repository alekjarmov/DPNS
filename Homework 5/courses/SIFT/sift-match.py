import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
mpl_use('TkAgg')

img1 = cv2.imread('../images/Example1.png', 0)
img2 = cv2.imread('../images/Example2.png', 0)

sift = cv2.SIFT_create()
#sift = cv2.AKAZE_create()
#sift = cv2.BRISK_create()

# find the keypoints and descriptors with SIFT
(kps1, descs1) = sift.detectAndCompute(img1, None)
(kps2, descs2) = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(descs1, descs2, k=2)

#Find the 2 nearest neighbors for a given descriptor. Let d1 be the distance to the nearest neighbor and d2 be the distance to the next one. 
#In order to accept the nearest neighbor as a "match", d1/d2 ratio should be smaller than a given threshold. 
#The motivation behind this test is that we expect a good match to be much closer to the query feature than the second best match.
# Because if both features are similarly close to the query, we cannot decide which one is really the best one.
 
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good, None, flags=2)
plt.imshow(img3), plt.show()
cv2.imwrite('img3.jpg', img3)
