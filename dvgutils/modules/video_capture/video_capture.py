import cv2

from .file_video_capture import FileVideoCapture, FileVideoCaptureThreaded
from .camera_video_capture import CameraVideoCapture, CameraVideoCaptureThreaded
from .pi_camera_video_capture import PiCameraVideoCapture, PiCameraVideoCaptureThreaded
from .stream_video_capture import StreamVideoCapture, StreamVideoCaptureThreaded
from ..transform import Transform


class VideoCapture:
    def __init__(self, conf, threaded=True):
        self.conf = conf
        self.threaded = threaded if "threaded" not in conf else conf["threaded"]
        self.capture = conf["capture"]

        # Setup optional video frame transformation function (resizing, flipping, etc.)
        transform = Transform(conf["transform"]) if "transform" in conf else None

        if conf["capture"] == "file":
            file_conf = conf["file"]
            kwargs = {}
            if "start_frame" in file_conf:
                kwargs["start_frame"] = file_conf["start_frame"]
            if "end_frame" in file_conf:
                kwargs["end_frame"] = file_conf["end_frame"]
            if "api_preference" in file_conf:
                kwargs["api_preference"] = getattr(cv2, file_conf["api_preference"])

            self.cap = \
                FileVideoCaptureThreaded(src=file_conf["src"], transform=transform, **kwargs) if self.threaded else \
                FileVideoCapture(src=file_conf["src"], transform=transform, **kwargs)
        elif conf["capture"] == "camera":
            camera_conf = conf["camera"]
            kwargs = {}
            if "src" in camera_conf:
                kwargs["src"] = camera_conf["src"]
            if "fourcc" in camera_conf:
                kwargs["fourcc"] = camera_conf["fourcc"]
            if "resolution" in camera_conf:
                kwargs["resolution"] = camera_conf["resolution"]
            if "fps" in camera_conf:
                kwargs["fps"] = camera_conf["fps"]
            if "api_preference" in camera_conf:
                kwargs["api_preference"] = getattr(cv2, camera_conf["api_preference"])

            self.cap = \
                CameraVideoCaptureThreaded(transform=transform, **kwargs) if self.threaded else \
                CameraVideoCapture(transform=transform, **kwargs)
        elif conf["capture"] == "piCamera":
            camera_conf = conf["piCamera"]
            kwargs = {}
            if "resolution" in camera_conf:
                kwargs["resolution"] = camera_conf["resolution"]
            if "fps" in camera_conf:
                kwargs["fps"] = camera_conf["fps"]
            if "settings" in camera_conf:
                kwargs = {**kwargs, **camera_conf["settings"]}

            self.cap = \
                PiCameraVideoCaptureThreaded(transform=transform, **kwargs) if self.threaded else \
                PiCameraVideoCapture(transform=transform, **kwargs)
        elif conf["capture"] == "stream":
            stream_conf = conf["stream"]
            kwargs = {}
            if "api_preference" in stream_conf:
                kwargs["api_preference"] = getattr(cv2, stream_conf["api_preference"])

            self.cap = \
                StreamVideoCaptureThreaded(src=stream_conf["src"], transform=transform, **kwargs) if self.threaded else \
                StreamVideoCapture(src=conf["src"], transform=transform, **kwargs)
        else:
            raise RuntimeError(f"Unsupported capture type: {conf['capture']}")

    def open(self):
        return self.cap.open()

    def read(self):
        return self.cap.read()

    def close(self):
        self.cap.close()
