import logging
import time
from threading import Thread
from queue import Queue


class PiCameraVideoCapture:
    def __init__(self, src=0, resolution=(320, 240), fps=32, transform=None, **kwargs):
        self.logger = logging.getLogger(__name__)

        self.src = src
        self.resolution = resolution
        self.fps = fps
        self.transform = transform
        self._kwargs = kwargs

        self.cam = None
        self.raw = None
        self.cap = None

    def open(self):
        from picamera.array import PiRGBArray
        from picamera import PiCamera

        # Initialize the camera
        self.cam = PiCamera(self.src)

        # Set camera parameters
        self.cam.resolution = self.resolution
        self.cam.framerate = self.fps

        # set optional camera parameters (refer to PiCamera docs: https://picamera.readthedocs.io/)
        for (arg, value) in self._kwargs.items():
            setattr(self.cam, arg, value)

        # initialize the stream
        self.raw = PiRGBArray(self.cam, size=self.resolution)
        self.cap = self.cam.capture_continuous(self.raw, format="bgr", use_video_port=True)

        # Update capture properties regardless to the device constraints
        self.resolution = self.cam.resolution
        self.fps = self.cam.framerate

        self.logger.info(f"Capturing Pi camera: {self.src}")
        self.logger.info(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        self.logger.info(f"FPS: {self.fps}")

        return self

    def read(self):
        f = next(self.cap)
        frame = f.array

        # Clear the stream in preparation for the next frame
        self.raw.truncate(0)

        if self.transform:
            frame = self.transform(frame)

        return frame

    def close(self):
        # Release camera resources
        self.cap.close()
        self.raw.close()
        self.cam.close()


class PiCameraVideoCaptureThreaded(PiCameraVideoCapture):
    def __init__(self, src=0, resolution=(320, 240), fps=32, transform=None,
                 queue_size=5, name="PiCameraVideoCaptureThreaded", **kwargs):
        super().__init__(src, resolution, fps, transform, **kwargs)

        # Initialize the queue used to store frames read from the PiCamera
        self.queue = Queue(maxsize=queue_size)

        # Initialize thread
        self.thread = Thread(target=self.capture, args=(), name=name)
        self.thread.daemon = True
        self.stopped = None

    def open(self):
        super().open()

        # Start a thread to read frames from the PiCamera along with the boolean
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
