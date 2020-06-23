import cv2
import os


class SaveVideo:
    """Save video stream

    :param str dst: name of the output file/stream
    :param int fps: framerate of the created video stream
    :param str | None fourcc: 4-character code of codec used to compress the frames
    :param overwrite:
    """
    def __init__(self, dst, api_preference=cv2.CAP_ANY, fps=30, fourcc="MJPG", overwrite=False):
        # FOURCC codec param and file extension combination is a crucial part.
        # Read more on: https://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/
        # See: http://www.fourcc.org/codecs.php

        if not overwrite and os.path.isfile(dst) and os.path.exists(dst):
            raise FileExistsError(f"{dst} already exists!")

        dirname = os.path.dirname(os.path.abspath(dst))
        os.makedirs(dirname, exist_ok=True)

        self.dst = dst
        self.api_preference = api_preference
        self.fps = fps
        self.writer = None
        self.fourcc = fourcc

    def __call__(self, frame):
        self.save(frame)

    def save(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                filename=self.dst,
                apiPreference=self.api_preference,
                fourcc=cv2.VideoWriter_fourcc(*self.fourcc) if self.fourcc else 0,
                fps=self.fps,
                frameSize=(w, h),
                isColor=(frame.ndim == 3))

        self.writer.write(frame)

    def close(self):
        if self.writer:
            self.writer.release()
