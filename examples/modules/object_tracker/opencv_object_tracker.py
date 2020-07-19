import cv2

from .centroid_tracker import CentroidTracker


class OpencvObjectTracker:

    def __init__(self, max_disappeared=40, max_distance=80, tracker_type="kcf"):
        # Initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
        self.w = None
        self.h = None

        self.tracker_type = tracker_type
        # Initialize a dictionary that maps strings to their corresponding OpenCV object tracker implementations
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

        # Instantiate our centroid tracker, then initialize a list to store each of our dlib correlation trackers,
        # followed by a dictionary to map each unique object ID to a TrackableObject
        self.centroid_tracker = CentroidTracker(max_disappeared, max_distance)
        self.trackers = []
        self.object_tracks = {}

    def track(self, frame, object_locations):

        # If the frame dimensions are empty, set them
        if self.w is None or self.h is None:
            (self.h, self.w) = frame.shape[:2]

        # Initialize  our list of bounding box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        rects = []

        # Check to see if we should run a more computationally expensive object detection method to aid our tracker
        if object_locations:
            # Initialize our new set of object trackers
            self.trackers = []
            # Loop over the detections
            for detection in object_locations:
                (start_x, start_y, end_x, end_y, class_name, confidence) = detection

                # Grab the appropriate object tracker using our dictionary of OpenCV object tracker objects
                tracker = self.OPENCV_OBJECT_TRACKERS[self.tracker_type]()
                tracker.init(frame, (start_x, start_y, end_x - start_x, end_y - start_y))

                # Add the tracker to our list of trackers so we can utilize it during skip frames
                self.trackers.append(tracker)
        else:
            # Loop over the trackers
            for tracker in self.trackers:
                # Update the tracker and grab the updated position
                (success, box) = tracker.update(frame)

                # Check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]

                    # Unpack the position object
                    start_x = int(x)
                    start_y = int(y)
                    end_x = int(x + w)
                    end_y = int(y + h)

                    # Add the bounding box coordinates to the rectangles list
                    rects.append((start_x, start_y, end_x, end_y))

        # Use the centroid tracker to associate the (1) old object centroids with
        # (2) the newly computed object centroids
        objects, bbox_dims = self.centroid_tracker.update(rects)
        current_objects = []

        # Loop over the tracked objects
        for (object_id, centroid) in objects.items():
            # Check to see if a trackable object exists for the current object ID
            to = self.object_tracks.get(object_id, None)

            # If there is no existing trackable object, create one
            if to is None:
                to = {
                    "object_id": object_id,
                    "centroids": [centroid],
                    "bbox_dims": bbox_dims[object_id],
                }
            # Otherwise, there is a trackable object so we can update it
            else:
                to["centroids"].append(centroid)

            current_objects.append(to)

            # Store the trackable object in our dictionary
            self.object_tracks[object_id] = to

        return current_objects
