import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import use as mpl_use
import os

mpl_use('TkAgg')

RESIZE_FACTOR = 0.15


def load_database() -> list[np.ndarray]:
    data_base_content = os.listdir("database/")
    database_locations = ["database/" + name for name in data_base_content]
    database_images = [cv2.imread(name, 0) for name in database_locations]
    # resize the images 1/5 of the original size
    database_images = [cv2.resize(img, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR) for img in database_images]
    return database_images


def load_query(image_index: int = 0) -> np.ndarray:
    query_images = os.listdir("query/")
    query_images = ["query/" + name for name in query_images]
    return cv2.imread(query_images[image_index], 0)


def load_query_colorized(image_index: int = 0) -> np.ndarray:
    query_images = os.listdir("query/")
    query_images = ["query/" + name for name in query_images]
    return cv2.cvtColor(cv2.imread(query_images[image_index]), cv2.COLOR_BGR2RGB)


def load_database_colorized(image_index: int = 0) -> np.ndarray:
    data_base_content = os.listdir("database/")
    database_locations = ["database/" + name for name in data_base_content]
    img = cv2.cvtColor(cv2.imread(database_locations[image_index]), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)


def find_image_with_most_inliers(img1, image_list):
    # Initialize variables to keep track of the maximum inliers
    max_inliers = 0
    img_best = None
    good_best = []
    i_best = 0
    keypoints_best, descriptors_best = None, None

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)

    for i, img2 in enumerate(image_list):

        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good_matches.append(m)

        # Extract matching keypoints from both images
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Count the number of inliers
        inliers = np.sum(mask)

        if inliers > max_inliers:
            max_inliers = inliers
            img_best = img2
            good_best = good_matches
            keypoints_best, descriptors_best = keypoints2, descriptors2
            i_best = i
            print("New best image found with {} inliers".format(max_inliers))
            print(i)

    return img_best, keypoints1, keypoints_best, good_best, i_best, descriptors1, descriptors_best


def main():
    # query_img_index = 1
    database_images = load_database()
    
    for query_img_index in range(2, -1, -1):
        img1 = load_query(query_img_index)


        img_best, keypoints1, keypoints_best, good_best, i_best, descriptors1, descriptors_best = find_image_with_most_inliers(
            img1, database_images)

        # cv2.imshow("Best matching image", img_best)
        img1 = load_query_colorized(query_img_index)
        img_best = load_database_colorized(i_best)

        plt.subplot(121), plt.imshow(img1, cmap='gray'), plt.title('Query image')
        plt.subplot(122), plt.imshow(img_best, cmap='gray'), plt.title('Best matching image')
        plt.show()


        img1_keypointed = cv2.drawKeypoints(img1, keypoints1, None, color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img_best_keypointed = cv2.drawKeypoints(img_best, keypoints_best, None, color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.subplot(121), plt.imshow(img1_keypointed, cmap='gray'), plt.title('Query image')
        plt.subplot(122), plt.imshow(img_best_keypointed, cmap='gray'), plt.title('Best matching image')
        plt.show()

        img3 = cv2.drawMatches(img1, keypoints1, img_best, keypoints_best, good_best, None, flags=2, matchColor=(255, 255, 0))
        plt.imshow(img3), plt.show()


        matches = [[m] for m in good_best] #cv2.DMatch object]
        img3 = cv2.drawMatchesKnn(img1, keypoints1, img_best, keypoints_best, matches, None, flags=2, matchColor=(255, 255, 0))
        plt.imshow(img3), plt.show()



if __name__ == "__main__":
    main()
