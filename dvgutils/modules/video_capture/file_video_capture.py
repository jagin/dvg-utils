import logging
import time
from threading import Thread
from queue import Queue

import cv2

from ...misc import decode_fourcc, str_to_sec


class FileVideoCapture:
    """Capture video from a file.

    :param str src: path to the source video file
    :param int | None api_preference: preferred Capture API backends to use
    :param int | str start_frame: frame/time to start from
    :param int | str end_frame: frame/time to end
    :param modules.transform.Transform | None transform: transformation callable class
    """
    def __init__(self, src, api_preference=None, start_frame=None, end_frame=None, transform=None):
        self.logger = logging.getLogger(__name__)

        self.src = src
        self.api_preference = api_preference
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.transform = transform
        self.cap = None

        self.fourcc = None
        self.resolution = None
        self.fps = None
        self.frame_count = None

    def open(self):
        """Open video file for video capturing.

        :returns: self

        :raises IOError: if cannot open video file
        """
        if self.api_preference:
            self.cap = cv2.VideoCapture(self.src, self.api_preference)
        else:
            self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.src}")

        # Get capture properties
        self.fourcc = decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.resolution = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.frame_count > 0:  # Sometimes OpenCV is not able to provide the length of the video
            # Set frame range
            if self.start_frame is None:
                self.start_frame = 1
            elif isinstance(self.start_frame, str):
                self.start_frame = int(str_to_sec(self.start_frame) * self.fps)
            if self.end_frame is None:
                self.end_frame = self.frame_count
            if isinstance(self.end_frame, str):
                self.end_frame = int(str_to_sec(self.end_frame) * self.fps)
            # Check frame range
            if not 1 <= self.start_frame < self.frame_count:
                self.logger.warning(f"Start frame {self.start_frame} out of range (1, {self.frame_count - 1})")
                self.logger.warning("Resetting start frame to 1")
                self.start_frame = 1
            if not 1 < self.end_frame <= self.frame_count:
                self.logger.warning(f"End frame {self.end_frame} out of range (1,{self.frame_count})")
                self.logger.warning(f"Resetting end frame reset to {self.frame_count}")
                self.end_frame = self.frame_count  # reset end_frame to frame_count

            # Set frame starting point for video capturing
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame - 1)

        self.logger.info(f"Capturing file: {self.src}")
        self.logger.info(f"Codec: {self.fourcc}")
        self.logger.info(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        self.logger.info(f"FPS: {self.fps}")

        return self

    def read(self):
        """Grabs, decodes and returns the next video frame.

        :returns: frame data or None if no frames left in the video stream
        :rtype: numpy.ndarray | None
        """
        (grabbed, frame) = self.cap.read()
        if grabbed and (self.frame_count < 0 or int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) <= self.end_frame):
            if self.transform:
                frame = self.transform(frame)

            return frame
        else:
            return None

    def __len__(self):
        return self.frame_count

    def close(self):
        """Close video file"""
        self.cap.release()


class FileVideoCaptureThreaded(FileVideoCapture):
    """Capture video from a file.

    :param str src: path to the source video file
    :param int | None api_preference: preferred Capture API backends to use
    :param int | str start_frame: frame/time to start from
    :param int | str end_frame: frame/time to end
    :param modules.transform.Transform | None transform: transformation callable class
    :param int queue_size: queue size to buffer video frames
    :param str name: thread name
    """
    def __init__(self, src, api_preference=None, start_frame=None, end_frame=None, transform=None,
                 queue_size=16, name="FileVideoCaptureThreaded"):
        super().__init__(src, api_preference, start_frame, end_frame, transform)

        # Initialize the queue used to store frames read from the video file
        self.queue = Queue(maxsize=queue_size)

        # Initialize thread
        self.thread = Thread(target=self.capture, args=(), name=name)
        self.thread.daemon = True
        self.stopped = None

    def open(self):
        super().open()

        # Start a thread to read frames from the video file along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stopped = False
        self.thread.start()

        return self

    def capture(self):
        while not self.stopped:
            if not self.queue.full():
                # Grab the frame from the video stream
                frame = super().read()

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
