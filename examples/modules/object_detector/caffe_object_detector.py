import numpy as np
import cv2

from dvgutils.vis import resize


class CaffeObjectDetector:

    def __init__(self, prototxt, model, frame_skip=None, confidence=0.3, wanted_classes=["person"]):
        self.detected_frames = 0
        self.frame_skip = frame_skip

        self.confidence = confidence
        self.wanted_classes = wanted_classes

        # load our serialized model from disk
        self.detector = cv2.dnn.readNetFromCaffe(prototxt, model)

        # initialize the list of class labels MobileNet SSD was trained to
        # detect
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.w = None
        self.h = None

    def preprocess(self, image):
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        image_blob = cv2.dnn.blobFromImage(image, 0.007843, image.shape[:2][::-1], 127.5)

        return image_blob

    def detect(self, image):

        if self.frame_skip is None or self.detected_frames % (self.frame_skip + 1) == 0:
            # if the frame dimensions are empty, set them
            if self.w is None or self.h is None:
                (self.h, self.w) = image.shape[:2]

            # resize the frame to have a chosen max width pixels (the
            # less data we have, the faster we can process it)
            frame = resize(image, 500)

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            image_blob = self.preprocess(frame)
            self.detector.setInput(image_blob)
            detections = self.detector.forward()

            # instantiate our coordinates array
            locations = []

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > self.confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if self.CLASSES[idx] not in self.wanted_classes:
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    locations.append(
                        (start_x, start_y, end_x, end_y,
                         self.CLASSES[idx], confidence))
        else:
            locations = None

        self.detected_frames += 1

        return locations
