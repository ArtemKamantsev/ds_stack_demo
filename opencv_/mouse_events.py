"""To draw circle you need:
1. press and hold the 'Alt' key
2. press and hold mouse left button
3. now you could release 'Alt' key
4. move mouse
5. release mouse left button
"""

import cv2
import numpy as np

IMG_ORIGINAL: np.ndarray = np.zeros((512, 512, 3), np.uint8)
IMG_TO_SHOW: np.ndarray = IMG_ORIGINAL.copy()

START_POINT: tuple[int, int] | None = None


def draw(img_to_draw: np.ndarray,
         point_start: tuple[int, int],
         point_end: tuple[int, int],
         color: tuple[int, int, int] = (0, 255, 0),
         thickness: int = -1) -> None:
    center: tuple[int, int] = (point_start[0] + point_end[0]) // 2, (point_start[1] + point_end[1]) // 2
    radius: int = max(abs(point_start[0] - point_end[0]), abs(point_start[1] - point_end[1])) // 2
    cv2.circle(img_to_draw, center, radius, color, thickness)


def draw_circle(event: int, x: int, y: int, flags: int, _) -> None:
    # pylint: disable=global-statement
    global START_POINT, IMG_TO_SHOW
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_ALTKEY:
        START_POINT = x, y
    elif event == cv2.EVENT_LBUTTONUP and START_POINT is not None:
        draw(IMG_ORIGINAL, START_POINT, (x, y))
        IMG_TO_SHOW = IMG_ORIGINAL.copy()
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and START_POINT is not None:
        IMG_TO_SHOW = IMG_ORIGINAL.copy()
        draw(IMG_TO_SHOW, START_POINT, (x, y))


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while True:
    cv2.imshow('image', IMG_TO_SHOW)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
