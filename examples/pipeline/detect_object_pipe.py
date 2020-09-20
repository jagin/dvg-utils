from modules.object_detector import ObjectDetector


class DetectObjectPipe:

    def __init__(self, conf):
        self.detector = ObjectDetector(conf)

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        image = data["image"]

        # Detect objects
        data["object_locations"] = self.detector.detect(image)

        return data
