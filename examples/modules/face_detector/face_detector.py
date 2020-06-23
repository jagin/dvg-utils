class FaceDetector:
    def __init__(self, conf):
        if conf["model"] == "cascade":
            from .cascade_face_detector import CascadeFaceDetector
            self.detector = CascadeFaceDetector(**conf["cascade"])
        elif conf["model"] == "caffe":
            from .caffe_face_detector import CaffeFaceDetector
            self.detector = CaffeFaceDetector(**conf["caffe"])
        else:
            raise RuntimeError(f"FaceDetector not initialized. Unknown model {conf['model']}!")

    def detect(self, image):
        return self.detector.detect(image)
