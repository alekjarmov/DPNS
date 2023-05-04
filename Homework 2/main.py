import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


kernels = np.array([
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[1, 1, 0], [1, 0, -1], [0, -1, -1]],
    [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
    [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    [[0, -1, -1], [1, 0, -1], [1, 1, 0]],
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]
])
names = ["Identity", "North West", "North", "North East", "West", "East", "South West", "South", "South East"]
image = cv2.imread("image_original.jpg")

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
filtered = []
for i, kernel in enumerate(kernels):

    image = cv2.filter2D(cv2.imread("image_original.jpg"), -1, kernel)
    if i!= 0:
        filtered.append(image)
    axs[i // 3, i % 3].imshow(image)
    axs[i // 3, i % 3].set_title(names[i])
    axs[i // 3, i % 3].axis('off')
plt.savefig('compass_output.jpg', bbox_inches='tight')
fig.show()

# adding the combined filters
combined_filters = np.maximum.reduce(filtered)
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
axs[0].imshow(cv2.imread("image_original.jpg"))
axs[0].set_title("Original")
axs[0].axis('off')
image = cv2.imread("image_original.jpg")
image = cv2.filter2D(image, -1, combined_filters)
axs[1].imshow(combined_filters)
axs[1].set_title("Combined filters")
axs[1].axis('off')
plt.savefig('combined_output.jpg', bbox_inches='tight')
fig.show()