from modules.object_tracker import ObjectTracker


class TrackObjectPipe:

    def __init__(self, conf):
        self.tracker = ObjectTracker(conf)

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        image = data["image"]
        object_locations = data["object_locations"]

        # Detect faces
        data["tracked_objects"] = self.tracker.track(image, object_locations)

        return data
