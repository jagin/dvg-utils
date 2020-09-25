import cv2

import dlib
from .centroid_tracker import CentroidTracker


class DlibObjectTracker:
    def __init__(self, max_disappeared=20, max_distance=80):
        # Initialize the frame dimensions (we'll set them as soon as we read the first frame from the video)
        self.w = None
        self.h = None

        # Instantiate our centroid tracker, then initialize a list to store each of our dlib correlation trackers,
        # followed by a dictionary to map each unique object ID to a TrackableObject
        self.centroid_tracker = CentroidTracker(max_disappeared, max_distance)
        self.trackers = []
        self.object_tracks = {}

    def track(self, frame, object_locations):
        # Convert the frame from BGR to RGB for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # If the frame dimensions are empty, set them
        if self.w is None or self.h is None:
            (self.h, self.w) = frame.shape[:2]

        # Initialize  our list of bounding box rectangles returned by either
        # (1) our object detector or
        # (2) the correlation trackers
        rects = []

        # Check to see if there are detected object locations from object detector to aid our tracker
        if object_locations:
            # Initialize our new set of object trackers
            self.trackers = []

            # Loop over the detections
            for detection in object_locations:
                (start_x, start_y, end_x, end_y) = detection[0:4]

                # Construct a dlib rectangle object from the bounding box coordinates and
                # then start the dlib correlation tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                tracker.start_track(rgb, rect)

                # Add the tracker to our list of trackers so we can utilize it during skip frames
                self.trackers.append(tracker)
        else:
            # Loop over the trackers
            for tracker in self.trackers:
                # Update the tracker and grab the updated position
                confidence = tracker.update(rgb)
                pos = tracker.get_position()

                # Unpack the position object
                start_x = int(pos.left())
                start_y = int(pos.top())
                end_x = int(pos.right())
                end_y = int(pos.bottom())

                # Add the bounding box coordinates to the rectangles list
                rects.append((start_x, start_y, end_x, end_y))

        # Use the centroid tracker to associate the
        # (1) old object centroids with
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
                    "bbox_dims": bbox_dims[object_id]
                }
            # Otherwise, there is a trackable object so we can update it
            else:
                to["centroids"].append(centroid)

            current_objects.append(to)

            # Store the trackable object in our dictionary
            self.object_tracks[object_id] = to

        return current_objects
