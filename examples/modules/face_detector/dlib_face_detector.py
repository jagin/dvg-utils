import numpy as np
import cv2
import dlib


class DlibFaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        detections = self.detector(image, 1)
        locations = []

        for detection in detections:
            start_x = detection.left()  # left point
            start_y = detection.top()  # top point
            end_x = detection.right()  # right point
            end_y = detection.bottom()  # bottom point
            locations.append((start_x, start_y, end_x, end_y))

        return np.array(locations, dtype=int)
