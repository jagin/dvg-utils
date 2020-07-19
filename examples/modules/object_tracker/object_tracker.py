class ObjectTracker:
    def __init__(self, conf):
        if conf["tracker"] == "dlib":
            from .dlib_object_tracker import DlibObjectTracker
            self.tracker = DlibObjectTracker(conf["max_disappeared"], conf["max_distance"])
        elif conf["tracker"] == "opencv":
            from .opencv_object_tracker import OpencvObjectTracker
            self.tracker = OpencvObjectTracker(conf["max_disappeared"], conf["max_distance"],
                                               conf["opencv"]["tracker_type"])
        else:
            raise RuntimeError(f"ObjectCounter not initialized. Unknown tracker {conf['tracker']}!")

    def track(self, image, detected_object_locations):
        return self.tracker.track(image, detected_object_locations)
