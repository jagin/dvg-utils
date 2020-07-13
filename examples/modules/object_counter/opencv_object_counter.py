import numpy as np
import cv2

import dlib
from .centroid_tracker import CentroidTracker
from .trackable_object import TrackableObject


class OpencvObjectCounter:

    def __init__(self, max_disappeared=40, max_distance=80, tracker_type="kcf"):

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.w = None
        self.h = None

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.total_frames = 0
        self.total_up = 0
        self.total_down = 0

        self.tracker_type = tracker_type
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(max_disappeared, max_distance)
        self.trackers = []
        self.trackable_objects = {}

    def detect(self, frame, detected_object_locations):

        # if the frame dimensions are empty, set them
        if self.w is None or self.h is None:
            (self.h, self.w) = frame.shape[:2]

        # initialize  our list of bounding box rectangles returned
        # by either (1) our object detector or
        # (2) the correlation trackers
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if detected_object_locations is not None:
            # initialize our new set of object trackers
            self.trackers = []

            detections = detected_object_locations

            # loop over the detections
            for detection in detections:
                (start_x, start_y, end_x, end_y, class_name, confidence) = detection

                # grab the appropriate object tracker using our dictionary of
                # OpenCV object tracker objects
                tracker = self.OPENCV_OBJECT_TRACKERS[self.tracker_type]()
                tracker.init(frame, (start_x, start_y, end_x-start_x, end_y-start_y))

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                self.trackers.append(tracker)
        else:
            # loop over the trackers
            for tracker in self.trackers:
                # update the tracker and grab the updated position
                (success, box) = tracker.update(frame)

                # check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]

                    # unpack the position object
                    start_x = int(x)
                    start_y = int(y)
                    end_x = int(x + w)
                    end_y = int(y + h)

                    # add the bounding box coordinates to the rectangles list
                    rects.append((start_x, start_y, end_x, end_y))

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = self.ct.update(rects)
        objects_coords = {}

        # loop over the tracked objects
        for (object_id, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackable_objects.get(object_id, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(object_id, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < self.h // 2:
                        self.total_up += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > self.h // 2:
                        self.total_down += 1
                        to.counted = True

            objects_coords[object_id] = (centroid[0], centroid[1])

            # store the trackable object in our dictionary
            self.trackable_objects[object_id] = to

        self.total_frames += 1

        return objects_coords, (self.total_up, self.total_down)
