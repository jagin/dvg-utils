import logging

import cv2

from dvgutils import setup_logger, load_config, colors
from dvgutils.vis import put_text, rectangle_overlay
from dvgutils.modules import VideoCapture, ShowImage, Metrics, SaveVideo, Progress
from dvgutils.pipeline import CaptureVideoPipe, MetricsPipe, Pipeline, ShowImagePipe, SaveVideoPipe, ProgressPipe

from modules.object_detector import ObjectDetector
from pipeline.detect_object_pipe import DetectObjectPipe
from pipeline.visualize_data_pipe import VisualizeDataPipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/object_detector_config.yml",
                        help="Path to the input configuration file (default: config/config.yml)")
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


def detect_face(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup processing modules
    video_capture = VideoCapture(conf["videoCapture"]).open()
    object_detector = ObjectDetector(conf["objectDetector"])
    video_fps = args["fps"] if args["fps"] is not None else video_capture.fps
    save_video = SaveVideo(args["output"], fps=video_fps) if args["output"] else None
    show_image = ShowImage("Video") if args["display"] else None
    metrics = Metrics().start()
    progress = Progress(disable=not args["progress"])

    try:
        logger.info("Capturing...")
        while True:
            # Grab the frame
            frame = video_capture.read()
            if frame is None:
                break

            # Detect objects
            object_locations = object_detector.detect(frame)

            # Visualize data
            #
            # FPS
            put_text(frame, f"{metrics.iter_per_sec():.2f} fps", (2, 2), org_pos="tl",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)
            # Frame number
            h, w = frame.shape[:2]
            put_text(frame, f"frame: {len(metrics)}", (w - 2, 2), org_pos="tr",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)
            # Face bounding boxes
            if object_locations:
                for object_location in object_locations:
                    (start_x, start_y, end_x, end_y, label, confidence) = object_location
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
                    rectangle_overlay(frame, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)
                    put_text(frame, str(label), (start_x - 10, start_y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    put_text(frame, str(confidence), (start_x - 10, start_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if save_video:
                save_video(frame)

            if show_image:
                show = show_image(frame)
                if not show:
                    break

            metrics.update()
            progress.update()

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


def detect_face_pipeline(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup pipeline steps
    capture_video_pipe = CaptureVideoPipe(conf["videoCapture"])
    visualize_data_pipe = VisualizeDataPipe("vis_image", {
        "metrics": True,
        "detected_object_locations": True
    })
    detect_object_pipe = DetectObjectPipe(conf["objectDetector"])
    video_fps = args["fps"] if args["fps"] is not None else capture_video_pipe.video_capture.fps
    save_video_pipe = SaveVideoPipe("vis_image", args["output"], fps=video_fps) if args["output"] else None
    show_image_pipe = ShowImagePipe("vis_image", "Video") if args["display"] else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe(disable=not args["progress"])

    # Create pipeline
    pipeline = Pipeline(capture_video_pipe)
    pipeline.map(detect_object_pipe)
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
        detect_face_pipeline(args)
    else:
        detect_face(args)
