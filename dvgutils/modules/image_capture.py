import os
import cv2

from ..fs import list_files
from .transform import Transform


class ImageCapture:
    def __init__(self, conf):
        self.conf = conf
        self.path = conf["path"]
        self.valid_ext = conf["valid_ext"]
        self.contains = conf["contains"] if "contains" in conf else None
        self.level = conf["level"] if "level" in conf else None

        # Setup optional image transformation function (resizing, flipping, etc.)
        self.transform = Transform(conf["transform"]) if "transform" in conf else None

        if os.path.isfile(self.path):
            self.source = iter([self.path])
        else:
            self.source = list_files(self.path, self.valid_ext, self.contains, self.level)

    def read(self):
        try:
            filename = next(self.source)
            image = cv2.imread(filename)
            if self.transform:
                image = self.transform(image)
            return filename, image
        except StopIteration:
            return None, None

