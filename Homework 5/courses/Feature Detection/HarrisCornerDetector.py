import cv2
import numpy as np

filename = '../images/example.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

#img - Input image, it should be grayscale and float32 type.
#blockSize - It is the size of neighbourhood considered for corner detection
#ksize - Aperture parameter of Sobel derivative used.
#k - Harris detector free parameter in the equation.

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
cv2.imshow('dst1', dst)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01*dst.max()] = [0, 0, 255]

cv2.imshow('dst2', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()