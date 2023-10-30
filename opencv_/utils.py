import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(img: np.ndarray, dpi=100, title=None) -> None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(dpi=dpi)
    plt.imshow(img_rgb)
    if title is not None:
        plt.title(title)
    plt.show()
