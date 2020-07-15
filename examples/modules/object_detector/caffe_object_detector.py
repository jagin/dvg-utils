import numpy as np
import cv2

from dvgutils.vis import resize


class CaffeObjectDetector:

    def __init__(self, prototxt, model, classes, frame_skip=None, confidence=0.5):
        self.iteration = 0
        self.frame_skip = frame_skip

        self.confidence = confidence
        self.classes = classes

        # Load our serialized model from disk
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, model)

        # Initialize the list of class labels MobileNet SSD was trained to detect
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # Initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
        self.w = None
        self.h = None

    def preprocess(self, image):
        # Convert the frame to a blob
        image_blob = cv2.dnn.blobFromImage(image, 0.007843, image.shape[:2][::-1], 127.5)

        return image_blob

    def detect(self, image):
        # Instantiate our coordinates array
        locations = []

        # If the frame dimensions are empty, set them
        if self.w is None or self.h is None:
            (self.h, self.w) = image.shape[:2]

        if self.frame_skip is None or self.iteration % (self.frame_skip + 1) == 0:
            # Resize the frame to have a chosen max width pixels
            # (the less data we have, the faster we can process it)
            frame = resize(image, 500)

            # Convert the frame to a blob and pass the blob through the network and obtain the detections
            image_blob = self.preprocess(frame)
            # and pass the blob through the network to obtain the detections
            self.detector.setInput(image_blob)
            detections = self.detector.forward()

            # Loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # Extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by requiring a minimum confidence
                if confidence > self.confidence:
                    # Extract the index of the class label from the detections list
                    idx = int(detections[0, 0, i, 1])

                    # If the class label is not a person, ignore it
                    if self.CLASSES[idx] not in self.classes:
                        continue

                    # Compute the (x, y)-coordinates of the bounding box for the object
                    box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    locations.append((start_x, start_y, end_x, end_y, self.CLASSES[idx], confidence))

        self.iteration += 1

        return locations
