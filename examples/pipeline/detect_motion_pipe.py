from modules.motion_detector import MotionDetector


class DetectMotionPipe:

    def __init__(self, conf):
        self.detector = MotionDetector(**conf)

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        image = data["image"]

        # Detect faces
        data["occupied"], data["motion_locations"] = self.detector.detect(image)

        return data
