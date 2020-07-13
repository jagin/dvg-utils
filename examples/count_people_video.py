import logging

import cv2

from dvgutils import setup_logger, load_config, colors
from dvgutils.vis import put_text, rectangle_overlay
from dvgutils.modules import VideoCapture, ShowImage, Metrics, SaveVideo, Progress
from dvgutils.pipeline import CaptureVideoPipe, MetricsPipe, Pipeline, ShowImagePipe, SaveVideoPipe, ProgressPipe

from modules.object_counter import ObjectCounter
from modules.object_detector import ObjectDetector
from pipeline.visualize_data_pipe import VisualizeDataPipe
from pipeline.count_object_pipe import CountObjectPipe
from pipeline.detect_object_pipe import DetectObjectPipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/object_counter_config.yml",
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


def count_object(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup processing modules
    video_capture = VideoCapture(conf["videoCapture"]).open()
    object_detector = ObjectDetector(conf["objectDetector"])
    object_counter = ObjectCounter(conf["objectCounter"])
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
            else:
                (h, w) = frame.shape[:2]

            # Detect objects
            object_locations = object_detector.detect(frame)

            # Detect people
            people_objects, (total_up, total_down) = object_counter.detect(frame, object_locations)

            # Visualize data
            #
            # FPS
            put_text(frame, f"{metrics.iter_per_sec():.2f} fps", (2, 2), org_pos="tl",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)
            # Frame number
            put_text(frame, f"frame: {len(metrics)}", (w - 2, 2), org_pos="tr",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)

            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 2)

            # loop over the tracked objects
            for objectID in people_objects:
                (x_coord, y_coord) = people_objects[objectID]

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "Id {}".format(objectID)
                put_text(frame, text, (x_coord - 10, y_coord - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (x_coord, y_coord), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Up", total_up),
                ("Down", total_down),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                put_text(frame, text, (10, h - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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


def count_object_pipeline(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup pipeline steps
    capture_video_pipe = CaptureVideoPipe(conf["videoCapture"])
    visualize_data_pipe = VisualizeDataPipe("vis_image", {
        "metrics": True,
        "tracked_object_locations": True
    })
    object_detector_pipe = DetectObjectPipe(conf["objectDetector"])
    count_object_pipe = CountObjectPipe(conf["objectCounter"])
    video_fps = args["fps"] if args["fps"] is not None else capture_video_pipe.video_capture.fps
    save_video_pipe = SaveVideoPipe("vis_image", args["output"], fps=video_fps) if args["output"] else None
    show_image_pipe = ShowImagePipe("vis_image", "Video") if args["display"] else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe(disable=not args["progress"])

    # Create pipeline
    pipeline = Pipeline(capture_video_pipe)
    pipeline.map(object_detector_pipe)
    pipeline.map(count_object_pipe)
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
        count_object_pipeline(args)
    else:
        count_object(args)
