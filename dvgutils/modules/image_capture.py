import os
import cv2

from ..fs import list_files


class ImageCapture:

    def __init__(self, path, valid_exts=(".jpg", ".png"), contains=None, level=None):
        self.path = path
        self.valid_exts = valid_exts
        self.contains = contains
        self.level = level

        if os.path.isfile(self.path):
            self.source = iter([self.path])
        else:
            self.source = list_files(self.path, self.valid_exts, self.contains, self.level)

    def read(self):
        try:
            filename = next(self.source)
            image = cv2.imread(filename)
            return filename, image
        except StopIteration:
            return None, None
