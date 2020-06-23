import os

import cv2
import numpy as np

from dvgutils.vis import clip_points


class CascadeFaceDetector:
    def __init__(self, classifier, **kwargs):
        if not os.path.isfile(classifier):
            raise FileNotFoundError(f"Cascade classifier {classifier} does not exists!")

        self.detector = cv2.CascadeClassifier(classifier)
        self.kwargs = kwargs

    def detect(self, image):
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
                                               minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
                                               **self.kwargs)

        # OpenCV returns bounding box coordinates in (x, y, w, h)
        # order but we want them in (start_x, start_y, end_x, end_y)
        locations = np.array([(x, y, x + w, y + h) for (x, y, w, h) in rects], dtype=int)

        if len(locations) > 0:
            # Sometimes the values get negative, we need to clip them out
            h, w = image.shape[:2]
            locations[:, :2] = clip_points(locations[:, :2], w, h)
            locations[:, 2:4] = clip_points(locations[:, 2:4], w, h)

        return locations
