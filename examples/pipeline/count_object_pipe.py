from modules.object_counter import ObjectCounter


class CountObjectPipe:

    def __init__(self, conf):
        self.detector = ObjectCounter(conf)

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        image = data["image"]
        object_locations = data["object_locations"]

        # Detect faces
        data["tracked_object_locations"], data["object_count"] = self.detector.detect(image, object_locations)

        return data
