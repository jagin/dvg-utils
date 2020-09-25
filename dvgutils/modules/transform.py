import cv2
from ..vis import resize


class Transform:
    def __init__(self, conf):
        self.conf = conf

    def __call__(self, image):
        if "resize" in self.conf:
            image = resize(image, **self.conf["resize"])
        if "flip" in self.conf:
            image = cv2.flip(image, self.conf["flip"])

        return image
