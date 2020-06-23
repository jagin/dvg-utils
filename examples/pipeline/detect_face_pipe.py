from modules.face_detector import FaceDetector


class DetectFacePipe:

    def __init__(self, conf):
        self.detector = FaceDetector(conf)

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        image = data["image"]

        # Detect faces
        data["face_locations"] = self.detector.detect(image)

        return data
