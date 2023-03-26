import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

image = cv2.imread("image1.jpg", 0)
cv2.imshow('image', image)

image_5 = cv2.bitwise_and(image, 0b11111000)
image_4 = cv2.bitwise_and(image, 0b11110000)
image_3 = cv2.bitwise_and(image, 0b11100000)
image_2 = cv2.bitwise_and(image, 0b11000000)
image_1 = cv2.bitwise_and(image, 0b10000000)

images_dict = {"5 bits": image_5,
               "4 bits": image_4,
               "3 bits": image_3,
               "2 bits": image_2,
               "1 bits": image_1}

fig = plt.figure()
spec = mpl.gridspec.GridSpec(ncols=6, nrows=2)

ax1 = fig.add_subplot(spec[0, 0:2])
ax2 = fig.add_subplot(spec[0, 2:4])
ax3 = fig.add_subplot(spec[0, 4:])
ax4 = fig.add_subplot(spec[1, 1:3])
ax5 = fig.add_subplot(spec[1, 3:5])
axs = [ax1, ax2, ax3, ax4, ax5]

for i, (key, value) in enumerate(images_dict.items()):
    axs[i].imshow(value, cmap='gray', interpolation='bicubic')
    axs[i].set_title(key)
    axs[i].axis('off')
plt.savefig('output.jpg', bbox_inches='tight')
fig.show()
