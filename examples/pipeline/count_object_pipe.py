from modules.object_counter import ObjectCounter


class CountObjectPipe:

    def __init__(self, conf):
        self.counter = ObjectCounter(conf)

    def __call__(self, data):
        return self.detect(data)

    def detect(self, data):
        tracked_objects = data["tracked_objects"]

        # Count objects
        data["crossed_in_out"], data["line"] = self.counter.count(tracked_objects)

        return data
