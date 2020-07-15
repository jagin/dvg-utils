import logging
import time

from dvgutils import setup_logger, load_config, colors
from dvgutils.vis import put_text, rectangle_overlay
from dvgutils.modules import VideoCapture, ShowImage, Metrics, SaveVideo, Progress
from dvgutils.pipeline import CaptureVideoPipe, MetricsPipe, Pipeline, ShowImagePipe, SaveVideoPipe, ProgressPipe

from modules.motion_detector import MotionDetector
from pipeline.detect_motion_pipe import DetectMotionPipe

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/detect_motion_video.yml",
                        help="Path to the input configuration file (default: config/detect_motion_video.yml)")
    parser.add_argument("-o", "--output", type=str,
                        help="output video file name")
    parser.add_argument("--no-display", dest='display', action="store_false",
                        help="hide display window")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="don't display progress")
    parser.add_argument("--pipeline", action="store_true",
                        help="run as pipeline")
    parser.add_argument("--metrics", type=str,
                        help="metrics file name")
    parser.add_argument("--fps", type=int,
                        help="output video fps")

    return vars(parser.parse_args())


def visualize_frame_info(vis_image, frame_num, fps):
    h, w = vis_image.shape[:2]

    # Visualize frame number
    put_text(vis_image, f"{frame_num}", (w, h), org_pos="br",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)
    # Visualize FPS
    put_text(vis_image, f"{fps:.2f} fps", (0, h), org_pos="bl",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)


def visualize_motion_locations(vis_image, motion_locations):
    if motion_locations:
        for location in motion_locations:
            (x1, y1, x2, y2) = location
            rectangle_overlay(vis_image, (x1, y1), (x2, y2), colors.get("red").bgr(), 0.5)
        put_text(vis_image, "MOTION DETECTED!", (0, 0), org_pos="tl",
                 bg_color=colors.get("red").bgr())


class VisualizeDataPipe:
    def __init__(self, image_key="vis_image"):
        self.image_key = image_key

    def __call__(self, data):
        return self.visualize(data)

    def visualize(self, data):
        vis_image = data["image"].copy()
        data[self.image_key] = vis_image

        self.visualize_frame_info(data)
        self.visualize_motion_locations(data)

        return data

    def visualize_frame_info(self, data):
        vis_image = data[self.image_key]
        frame_num = data["idx"]
        fps = data["fps"]

        visualize_frame_info(vis_image, frame_num, fps)

    def visualize_motion_locations(self, data):
        vis_image = data[self.image_key]
        motion_locations = data["motion_locations"]

        visualize_motion_locations(vis_image, motion_locations)


def detect_motion(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup processing modules
    video_capture = VideoCapture(conf["videoCapture"]).open()
    motion_detector = MotionDetector(**conf["motionDetector"])
    video_fps = args["fps"] if args["fps"] is not None else video_capture.fps
    save_video = SaveVideo(args["output"], fps=video_fps) if args["output"] else None
    show_image = ShowImage("Video") if args["display"] else None
    metrics = Metrics().start()
    progress = Progress(disable=not args["progress"])

    try:
        logger.info("Capturing...")

        idx = 0
        start_time = time.perf_counter()
        while True:
            # Grab the frame
            frame = video_capture.read()
            if frame is None:
                break
            # Calculate FPS
            fps = (idx + 1) / (time.perf_counter() - start_time)

            # Detect motion
            locations = motion_detector.detect(frame)

            # Visualize data
            #
            # Metrics
            visualize_frame_info(frame, idx, fps)
            # Motion location
            visualize_motion_locations(frame, locations)

            # Save video
            if save_video:
                save_video(frame)

            # Show image
            if show_image:
                show = show_image(frame)
                if not show:
                    break

            # Update metrics
            metrics.update()
            # Update progress
            progress.update()

            idx += 1

        logger.info(f"{len(metrics)} it, "
                    f"{metrics.elapsed():.3f} s, "
                    f"{metrics.sec_per_iter():.3f} s/it, "
                    f"{metrics.iter_per_sec():.2f} it/s")

        if args["metrics"]:
            metrics.save(args["metrics"])
            logger.info(f"Metrics saved to {args['metrics']}")

    except KeyboardInterrupt:
        logger.warning("Got Ctrl+C!")
    finally:
        # Cleanup resources
        progress.close()
        if show_image:
            show_image.close()
        video_capture.close()


def detect_motion_pipeline(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup pipeline steps
    capture_video_pipe = CaptureVideoPipe(conf["videoCapture"])
    visualize_data_pipe = VisualizeDataPipe("vis_image")
    detect_motion_pipe = DetectMotionPipe(conf["motionDetector"])
    video_fps = args["fps"] if args["fps"] is not None else capture_video_pipe.video_capture.fps
    save_video_pipe = SaveVideoPipe("vis_image", args["output"], fps=video_fps) if args["output"] else None
    show_image_pipe = ShowImagePipe("vis_image", "Video") if args["display"] else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe(disable=not args["progress"])

    # Create pipeline
    pipeline = Pipeline(capture_video_pipe)
    pipeline.map(detect_motion_pipe)
    pipeline.map(visualize_data_pipe)
    pipeline.map(save_video_pipe)
    pipeline.map(show_image_pipe)
    pipeline.map(metrics_pipe)
    pipeline.map(progress_pipe)

    # Process pipeline
    try:
        logger.info("Capturing...")
        pipeline.run()

        logger.info(f"{len(metrics_pipe.metrics)} it, "
                    f"{metrics_pipe.metrics.elapsed():.3f} s, "
                    f"{metrics_pipe.metrics.sec_per_iter():.3f} s/it, "
                    f"{metrics_pipe.metrics.iter_per_sec():.2f} it/s")

        if args["metrics"]:
            metrics_pipe.metrics.save(args["metrics"])
            logger.info(f"Metrics saved to {args['metrics']}")
    except KeyboardInterrupt:
        logger.warning("Got Ctrl+C!")
    finally:
        # Cleanup pipeline resources
        pipeline.close()


if __name__ == "__main__":
    setup_logger()

    args = parse_args()
    if args["pipeline"]:
        detect_motion_pipeline(args)
    else:
        detect_motion(args)
