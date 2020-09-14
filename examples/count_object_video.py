import logging

from dvgutils import setup_logger, load_config
from dvgutils.pipeline import CaptureVideoPipe, MetricsPipe, Pipeline, ShowImagePipe, SaveVideoPipe, ProgressPipe

from utils.vis import visualize_frame_info, visualize_object_counter, visualize_tracked_object_locations
from pipeline.count_object_pipe import CountObjectPipe
from pipeline.track_object_pipe import TrackObjectPipe
from pipeline.detect_object_pipe import DetectObjectPipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/count_object_video.yml",
                        help="Path to the input configuration file (default: config/count_object_video.yml)")
    parser.add_argument("-cfo", "--conf-overwrites", nargs="+", type=str)
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


class VisualizeDataPipe:
    def __init__(self, image_key="vis_image"):
        self.image_key = image_key

    def __call__(self, data):
        return self.visualize(data)

    def visualize(self, data):
        vis_image = data["image"].copy()
        data[self.image_key] = vis_image

        self.visualize_frame_info(data)
        self.visualize_object_counter(data)
        self.visualize_tracked_object_locations(data)

        return data

    def visualize_frame_info(self, data):
        vis_image = data[self.image_key]
        frame_num = data["idx"]
        fps = data["fps"]

        visualize_frame_info(vis_image, frame_num, fps)

    def visualize_object_counter(self, data):
        vis_image = data[self.image_key]
        crossed_in_out = data["crossed_in_out"]
        lines = data["line"]

        visualize_object_counter(vis_image, crossed_in_out, lines)

    def visualize_tracked_object_locations(self, data):
        vis_image = data[self.image_key]
        tracked_objects = data["tracked_objects"]

        visualize_tracked_object_locations(vis_image, tracked_objects)


def count_object(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"], args["conf_overwrites"])

    # Setup pipeline steps
    capture_video_pipe = CaptureVideoPipe(conf["videoCapture"])
    visualize_data_pipe = VisualizeDataPipe("vis_image")
    object_detector_pipe = DetectObjectPipe(conf["objectDetector"])
    track_object_pipe = TrackObjectPipe(conf["objectTracker"])
    count_object_pipe = CountObjectPipe(conf["objectCounter"])
    video_fps = args["fps"] if args["fps"] is not None else capture_video_pipe.video_capture.fps
    save_video_pipe = SaveVideoPipe("vis_image", args["output"], fps=video_fps) if args["output"] else None
    show_image_pipe = ShowImagePipe("vis_image", "Video") if args["display"] else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe(disable=not args["progress"])

    # Create pipeline
    pipeline = Pipeline(capture_video_pipe)
    pipeline.map(object_detector_pipe)
    pipeline.map(track_object_pipe)
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
