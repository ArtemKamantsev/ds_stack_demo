import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(img: np.ndarray) -> None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(dpi=100)
    plt.imshow(img_rgb)
    plt.show()
