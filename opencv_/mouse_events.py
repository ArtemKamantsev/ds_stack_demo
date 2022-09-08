"""To draw circle you need:
1. press and hold the 'Alt' key
2. press and hold mouse left button
3. now you could release 'Alt' key
4. move mouse
5. release mouse left button
"""

import cv2
import numpy as np

img_original: np.ndarray = np.zeros((512, 512, 3), np.uint8)
img_to_show: np.ndarray = img_original.copy()

start_point: tuple[int, int] | None = None


def draw(img_to_draw: np.ndarray,
         point_start: tuple[int, int],
         point_end: tuple[int, int],
         color: tuple[int, int, int] = (0, 255, 0),
         thickness: int = -1) -> None:
    center: tuple[int, int] = (point_start[0] + point_end[0]) // 2, (point_start[1] + point_end[1]) // 2,
    radius: int = max(abs(point_start[0] - point_end[0]), abs(point_start[1] - point_end[1])) // 2
    cv2.circle(img_to_draw, center, radius, color, thickness)


def draw_circle(event: int, x: int, y: int, flags: int, _) -> None:
    global start_point, img_to_show
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_ALTKEY:
        start_point = x, y
    elif event == cv2.EVENT_LBUTTONUP and start_point is not None:
        draw(img_original, start_point, (x, y))
        img_to_show = img_original.copy()
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and start_point is not None:
        img_to_show = img_original.copy()
        draw(img_to_show, start_point, (x, y))


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while True:
    cv2.imshow('image', img_to_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
