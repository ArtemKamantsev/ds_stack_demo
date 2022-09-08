import cv2
import numpy as np

cap = cv2.VideoCapture('../images/puppies.avi')
while cap.isOpened():
    ret: bool
    frame: np.ndarray

    ret, frame = cap.read()
    if not ret:
        break

    gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
