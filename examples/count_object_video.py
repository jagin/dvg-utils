import logging
import cv2

from dvgutils import setup_logger, load_config, colors
from dvgutils.vis import put_text
from dvgutils.pipeline import CaptureVideoPipe, MetricsPipe, Pipeline, ShowImagePipe, SaveVideoPipe, ProgressPipe

from pipeline.count_object_pipe import CountObjectPipe
from pipeline.detect_object_pipe import DetectObjectPipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/count_object_video.yml",
                        help="Path to the input configuration file (default: config/count_object_video.yml)")
    parser.add_argument("-o", "--output", type=str,
                        help="output video file name")
    parser.add_argument("--no-display", dest='display', action="store_false",
                        help="hide display window")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="don't display progress")
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


def visualize_tracked_object_locations(vis_image, tracked_object_locations, object_count):
    h, w = vis_image.shape[:2]
    (objects_up, objects_down) = object_count

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(vis_image, (0, h // 2), (w, h // 2), colors.get("red").bgr(), 2)

    # loop over the tracked objects
    for object_id in tracked_object_locations:
        (x_coord, y_coord) = tracked_object_locations[object_id]

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "Id {}".format(object_id)
        put_text(vis_image, text, (x_coord, y_coord), org_pos="bl",
                 color=colors.get("white").bgr(), bg_color=colors.get("green").bgr(),
                 bg_alpha=0.5)
        cv2.circle(vis_image, (x_coord, y_coord), 4, colors.get("green").bgr(), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", objects_up),
        ("Down", objects_down),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        put_text(vis_image, text, (0, ((i * 22) + 20)), org_pos="bl",
                 bg_color=colors.get("white").bgr(), bg_alpha=0.5)


class VisualizeDataPipe:
    def __init__(self, image_key="vis_image"):
        self.image_key = image_key

    def __call__(self, data):
        return self.visualize(data)

    def visualize(self, data):
        vis_image = data["image"].copy()
        data[self.image_key] = vis_image

        self.visualize_frame_info(data)
        self.visualize_tracked_object_locations(data)

        return data

    def visualize_frame_info(self, data):
        vis_image = data[self.image_key]
        frame_num = data["idx"]
        fps = data["fps"]

        visualize_frame_info(vis_image, frame_num, fps)

    def visualize_tracked_object_locations(self, data):
        vis_image = data[self.image_key]
        tracked_object_locations = data["tracked_object_locations"]
        object_count = data["object_count"]

        visualize_tracked_object_locations(vis_image, tracked_object_locations, object_count)


def count_object(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup pipeline steps
    capture_video_pipe = CaptureVideoPipe(conf["videoCapture"])
    visualize_data_pipe = VisualizeDataPipe("vis_image")
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
    count_object(args)
