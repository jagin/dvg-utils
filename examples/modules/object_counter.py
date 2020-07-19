import math


class ObjectCounter:
    def __init__(self, conf):
        line = conf["line"]

        # Initialize total number of objects that have moved either in or out of the zone
        self.crossed_in = 0
        self.crossed_out = 0

        self.line = line
        self.bbox = None

        self.bbox = [min(line[0], line[2]), min(line[1], line[3]), abs(line[0] - line[2]),
                     abs(line[1] - line[3])]

    def count(self, tracked_objects):
        for tracked_object in tracked_objects:
            if "counted" not in tracked_object:
                tracked_object["counted"] = False
                tracked_object["is_in"] = None

            centroid = tracked_object["centroids"][-1]
            point = (centroid[0], centroid[1])

            if self.rect_contains(self.bbox, point):
                v1 = (self.line[2] - self.line[0], self.line[3] - self.line[1])
                v2 = (self.line[2] - point[0], self.line[3] - point[1])

                cross_prod = v1[0] * v2[1] - v1[1] * v2[0]

                # Check to see if the object has been counted or not
                if not tracked_object["counted"]:
                    if cross_prod < 0:
                        tracked_object["is_in"] = True
                    elif cross_prod >= 0:
                        tracked_object["is_in"] = False

                    tracked_object["counted"] = True

                else:
                    if tracked_object["is_in"] is True and cross_prod >= 0:
                        self.crossed_out += 1
                        tracked_object["is_in"] = False
                    if tracked_object["is_in"] is False and cross_prod < 0:
                        self.crossed_in += 1
                        tracked_object["is_in"] = True
        return (self.crossed_in, self.crossed_out), self.line

    def rect_contains(self, rect, pt):
        logic = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
        return logic
