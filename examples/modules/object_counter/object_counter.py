class ObjectCounter:
    def __init__(self, conf):
        if conf["tracker"] == "dlib":
            from .dlib_object_counter import DlibObjectCounter
            self.detector = DlibObjectCounter(conf["max_disappeared"], conf["max_distance"])
        elif conf["tracker"] == "opencv":
            from .opencv_object_counter import OpencvObjectCounter
            self.detector = OpencvObjectCounter(conf["max_disappeared"], conf["max_distance"],
                                                conf["opencv"]["tracker_type"])
        else:
            raise RuntimeError(f"ObjectCounter not initialized. Unknown tracker {conf['tracker']}!")

    def detect(self, image, detected_object_locations):
        return self.detector.detect(image, detected_object_locations)
