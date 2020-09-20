import logging
import time
from threading import Thread
from queue import Queue

import cv2


class StreamVideoCapture:
    """Capture video from a stream.

    :param int | str src: source of the stream
    :param int api_preference: preferred Capture API backends to use
    :param modules.transform.Transform | None transform: transformation callable class
    """
    def __init__(self, src, api_preference=cv2.CAP_ANY, transform=None):
        self.logger = logging.getLogger(__name__)

        self.src = src
        self.api_preference = api_preference
        self.transform = transform
        self.cap = None
        self.resolution = None
        self.fps = None

    def open(self):
        # Open video capture
        if self.api_preference:
            self.cap = cv2.VideoCapture(self.src, self.api_preference)
        else:
            self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video stream: {self.src}")

        # Get capture properties
        self.resolution = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.logger.info(f"Capturing stream: {self.src}")
        self.logger.info(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        self.logger.info(f"FPS: {self.fps}")

        return self

    def read(self):
        (grabbed, frame) = self.cap.read()

        if grabbed:
            if self.transform:
                frame = self.transform(frame)

            return frame
        else:
            return None

    def close(self):
        self.cap.release()


class StreamVideoCaptureThreaded(StreamVideoCapture):
    def __init__(self, src, api_preference=cv2.CAP_ANY, transform=None,
                 queue_size=5, name="StreamVideoCaptureThreaded"):
        super().__init__(src, api_preference, transform)

        # Initialize the queue used to store frames read from the video stream
        self.queue = Queue(maxsize=queue_size)

        self.thread = Thread(target=self.capture, args=(), name=name)
        self.thread.daemon = True
        self.stopped = None

    def open(self):
        super().open()

        # Start a thread to read frames from the video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stopped = False
        self.thread.start()

        return self

    def capture(self):
        while not self.stopped:
            # Grab the frame from the video stream
            frame = super().read()

            if not self.queue.full():
                # add the frames to the queue
                self.queue.put(frame)

                if frame is None:
                    break
            else:
                time.sleep(0.01)  # Rest for 1ms, we have a full queue

    def read(self):
        # return next frame in the queue
        return self.queue.get()

    def close(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()
        # Close the video capture
        super().close()
