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
