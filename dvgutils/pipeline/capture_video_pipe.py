from ..modules import VideoCapture
from .observable import observable


class CaptureVideoPipe:

    def __init__(self, conf, **kwargs):
        super().__init__()

        self.video_capture = VideoCapture(conf, **kwargs).open()
        self.stop = False
        observable.register("stop", self, self.on_stop)

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return len(self.video_capture)

    def on_stop(self):
        self.stop = True

    def generator(self):
        idx = 0
        while not self.stop:
            image = self.video_capture.read()
            if image is not None:
                data = {
                    "idx": idx,
                    "name": f"{idx:06d}",
                    "image": image
                }
                idx += 1
                yield data
            else:
                break

    def close(self):
        self.video_capture.close()
