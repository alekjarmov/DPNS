import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
# read al the images form input_images folder
image_names = os.listdir("input_images/")
image_locations = ["input_images/" + name for name in image_names]

for name, path in zip(image_names, image_locations):
    img = cv2.imread(path)
    img_orig = img.copy()
    img = cv2.medianBlur(img, 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(255-thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contoured_image = cv2.drawContours(np.zeros(thresh.shape, np.uint8), contours, -1, 255, 1)
    contoured_image = cv2.bitwise_not(contoured_image)
    fig, axs = plt.subplots(1, 3, figsize=(27, 9))
    axs[0].imshow(img_orig)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(thresh, cmap=mpl.cm.gray)
    axs[1].set_title("Segmented")
    axs[1].axis('off')

    axs[2].imshow(contoured_image, cmap=mpl.cm.gray)
    axs[2].set_title("Contoured")
    axs[2].axis('off')

    plt.savefig(f'output/{name}.jpg', bbox_inches='tight')
    # fig.show()
    plt.close(fig)
    print(f"Image {name} saved to output/{name}.jpg")

# Избраниот алгоритам точно ги сегментира сликите
