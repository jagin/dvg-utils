import cv2

from dvgutils.vis import put_text, rectangle_overlay
from dvgutils import colors
from dvgutils.pipeline import observable


class VisualizeDataPipe:
    def __init__(self, image_key="vis_image", conf={}):
        self.image_key = image_key
        self.conf = conf
        self.metrics = None
        observable.register("metrics", self, self.on_metrics)

    def on_metrics(self, metrics):
        self.metrics = metrics

    def __call__(self, data):
        return self.visualize(data)

    def visualize(self, data):
        vis_image = data["image"].copy()
        data[self.image_key] = vis_image

        if "metrics" in self.conf and self.conf["metrics"] and self.metrics:
            self.visualize_fps(data)
            self.visualize_frame_num(data)

        if "face_locations" in self.conf and self.conf["face_locations"]:
            self.visualize_face_locations(data)

        if "motion_locations" in self.conf and self.conf["motion_locations"]:
            self.visualize_motion_locations(data)

        if "detected_object_locations" in self.conf and self.conf["detected_object_locations"]:
            self.visualize_detected_object_locations(data)

        if "tracked_object_locations" in self.conf and self.conf["tracked_object_locations"]:
            self.visualize_tracked_object_locations(data)

        return data

    def visualize_fps(self, data):
        vis_image = data[self.image_key]
        fps = self.metrics["iter_per_sec"]

        put_text(vis_image, f"{fps:.2f} fps", (2, 2), org_pos="tl",
                 bg_color=colors.get("white").bgr(), bg_alpha=0.5)

    def visualize_frame_num(self, data):
        vis_image = data[self.image_key]
        h, w = vis_image.shape[:2]
        frame_num = self.metrics["iteration"]

        put_text(vis_image, f"frame: {frame_num}", (w - 2, 2), org_pos="tr",
                 bg_color=colors.get("white").bgr(), bg_alpha=0.5)

    def visualize_face_locations(self, data):
        vis_image = data[self.image_key]
        face_locations = data["face_locations"]

        for face_location in face_locations:
            (start_x, start_y, end_x, end_y) = face_location[0:4]
            cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
            rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)

<<<<<<< examples/pipeline/visualize_data_pipe.py
    def visualize_motion_locations(self, data):
        vis_image = data[self.image_key]
        occupied = data["occupied"]
        motion_locations = data["motion_locations"]

        # Visualize data
        #
        text = ""
        if occupied == 1:
            for location in motion_locations:
                (x1, y1, x2, y2) = location
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = "Occupied"

        else:
            text = "Empty"

        # draw the text and timestamp on the frame
        cv2.putText(vis_image, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

=======
    def visualize_detected_object_locations(self, data):
        vis_image = data[self.image_key]
        detected_object_locations = data["detected_object_locations"]
        if detected_object_locations:
            for detected_object_location in detected_object_locations:
                (start_x, start_y, end_x, end_y, label, confidence) = detected_object_location
                cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
                rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)
                put_text(vis_image, str(label), (start_x - 10, start_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                put_text(vis_image, str(confidence), (start_x - 10, start_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def visualize_tracked_object_locations(self, data):
        vis_image = data[self.image_key]
        h, w = vis_image.shape[:2]

        tracked_object_locations = data["tracked_object_locations"]
        (objects_up, objects_down) = data["object_count"]

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(vis_image, (0, h // 2), (w, h // 2), (0, 255, 255), 2)

        # loop over the tracked objects
        for object_id in tracked_object_locations:
            (x_coord, y_coord) = tracked_object_locations[object_id]

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "Id {}".format(object_id)
            put_text(vis_image, text, (x_coord - 10, y_coord - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(vis_image, (x_coord, y_coord), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", objects_up),
            ("Down", objects_down),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            put_text(vis_image, text, (10, h - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
>>>>>>> examples/pipeline/visualize_data_pipe.py
