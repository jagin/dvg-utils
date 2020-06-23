import numpy as np
import cv2

from dvgutils.misc import clip_points


class CaffeFaceDetector:

    def __init__(self, prototxt, model, confidence=0.5):
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, model)
        #self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        self.confidence = confidence

    def preprocess(self, image):
        # Construct an input blob for the image by resizing to a fixed 300x300 pixels
        # and then normalizing it
        image_blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300),
            # cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        return image_blob

    def detect(self, image):
        h, w = image.shape[:2]

        image_blob = self.preprocess(image)

        # Apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(image_blob)
        detections = self.detector.forward()
        locations = []

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")

                # top, right, bottom, left order + confidence
                locations.append((start_x, start_y, end_x, end_y, confidence))

        # Sometimes the values get negative, we need to clip them out
        locations = np.array(locations, dtype=int)
        if len(locations) > 0:
            locations[:, :2] = clip_points(locations[:, :2], w, h)
            locations[:, 2:4] = clip_points(locations[:, 2:4], w, h)

        return locations
