import cv2

img = cv2.imread('../images/example.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
#sift = cv2.AKAZE_create()
(kps, descs) = sift.detectAndCompute(gray, None)
img = cv2.drawKeypoints(gray, kps, img)
cv2.imshow('sift_keypoints', img)
cv2.imwrite('img1.jpg', img)
img = cv2.drawKeypoints(gray, kps, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift_keypoints-rich', img)
cv2.imwrite('img2.jpg', img)
descfile = open('sift-descriptors.txt', 'w')
for desc in descs:
    descfile.write("%s\n" % desc)
descfile.close()
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

