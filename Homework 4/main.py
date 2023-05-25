import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

image_names = os.listdir("database/")
image_locations = ["database/" + name for name in image_names]


def contour_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img_orig = img.copy()
    img = cv2.medianBlur(img, 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, segmented = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(cv2.bitwise_not(segmented), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contoured_image = cv2.drawContours(np.zeros(segmented.shape, np.uint8), contours, -1, 255, 1)
    # contoured_image = cv2.bitwise_not(contoured_image)

    return contoured_image


def get_similarities(query_image_name: str, contours: dict[str, np.ndarray]) -> list[np.ndarray]:
    path = "query/" + query_image_name
    img = cv2.imread(path)
    query_contoured = contour_image(path)
    similarities: dict[str, float] = {}
    for name, contour in contours.items():
        similarities.update({name: cv2.matchShapes(query_contoured, contour, cv2.CONTOURS_MATCH_I1, 0)})

    # sort by descending order
    similarities = dict(sorted(similarities.items(), key=lambda x: x[1]))

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"Most similar image: {list(similarities.keys())[0]}")
    axs[0].imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    axs[0].set_title("Queried image")
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(cv2.imread("database/" + list(similarities.keys())[0]), cv2.COLOR_BGR2RGB))
    axs[1].set_title("Most similar image")
    axs[1].axis('off')
    plt.savefig(f'output/{query_image_name}-{list(similarities.keys())[0]}.jpg', bbox_inches='tight')
    # fig.show()
    plt.close(fig)

    print(f"Query image: {query_image_name} with similarities: {list(similarities.values())}")
    return [cv2.imread("database/" + name) for name in similarities.keys()]


def main():
    contours = {name: contour_image(path) for name, path in zip(image_names, image_locations)}
    query_names = os.listdir("query/")
    for query_name in query_names:
        similarities = get_similarities(query_name, contours)
        # print(f'Query image: {query_name} with similarities: {similarities}')


if __name__ == "__main__":
    main()


# функцијата враќа листа од слики кои се најслични со query сликата
# тие може да се видат во променливата similarities иако не се користи
# во output директориумот се зачувуваат сликите кои се најслични со query сликата
# во главно наоѓа точни сличности но се јавува проблем кога query сликата е многу мала во димензии







