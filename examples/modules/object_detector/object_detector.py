class ObjectDetector:
    def __init__(self, conf):
        if conf["model"] == "caffe":
            from .caffe_object_detector import CaffeObjectDetector
            self.detector = CaffeObjectDetector(**conf["caffe"])
        else:
            raise RuntimeError(f"ObjectDetector not initialized. Unknown model {conf['model']}!")

    def detect(self, image):
        return self.detector.detect(image)
