import logging

from dvgutils import setup_logger, load_config, colors
from dvgutils.vis import put_text
from dvgutils.modules import VideoCapture, ShowImage, Metrics, SaveVideo, Progress
from dvgutils.pipeline import observable, Pipeline, \
    CaptureVideoPipe, ShowImagePipe, SaveVideoPipe, MetricsPipe, ProgressPipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/config.yml",
                        help="Path to the input configuration file (default: config/config.yml)")
    parser.add_argument("-o", "--output", type=str,
                        help="output video file name")
    parser.add_argument("--no-display", dest='display', action="store_false",
                        help="hide display window")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="don't display progress")
    parser.add_argument("--pipeline", action="store_true",
                        help="run as pipeline")
    parser.add_argument("--fps", type=int,
                        help="output video fps")

    return vars(parser.parse_args())


def capture_video(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup processing modules
    video_capture = VideoCapture(conf["videoCapture"]).open()
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

            # Visualize data
            #
            # FPS
            put_text(frame, f"{metrics.iter_per_sec():.2f} fps", (2, 2), org_pos="tl",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)
            # Frame number
            h, w = frame.shape[:2]
            put_text(frame, f"frame: {len(metrics)}", (w - 2, 2), org_pos="tr",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)

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
    except KeyboardInterrupt:
        logger.warning("Got Ctrl+C!")
    finally:
        # Cleanup resources
        progress.close()
        if show_image:
            show_image.close()
        video_capture.close()


class VisualizeDataPipe:
    def __init__(self, image_key="vis_image"):
        self.image_key = image_key
        self.metrics = None
        observable.register("metrics", self, self.on_metrics)

    def on_metrics(self, metrics):
        self.metrics = metrics

    def __call__(self, data):
        return self.visualize(data)

    def visualize(self, data):
        vis_image = data["image"].copy()
        data[self.image_key] = vis_image
        h, w = vis_image.shape[:2]

        if self.metrics:
            # Visualize FPS
            put_text(vis_image, f"{self.metrics['iter_per_sec']:.2f} fps", (2, 2), org_pos="tl",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)

            # Visualize frame number
            put_text(vis_image, f"frame: {self.metrics['iteration']}", (w - 2, 2), org_pos="tr",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)

        return data


def capture_video_pipeline(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup pipeline steps
    capture_video_pipe = CaptureVideoPipe(conf["videoCapture"])
    visualize_data_pipe = VisualizeDataPipe("vis_image")
    video_fps = args["fps"] if args["fps"] is not None else capture_video_pipe.video_capture.fps
    save_video_pipe = SaveVideoPipe("vis_image", args["output"], fps=video_fps) if args["output"] else None
    show_image_pipe = ShowImagePipe("vis_image", "Video") if args["display"] else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe(disable=not args["progress"])

    # Create pipeline
    pipeline = Pipeline(capture_video_pipe)
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
    except KeyboardInterrupt:
        logger.warning("Got Ctrl+C!")
    finally:
        # Cleanup pipeline resources
        pipeline.close()


if __name__ == "__main__":
    setup_logger()

    args = parse_args()
    if args["pipeline"]:
        capture_video_pipeline(args)
    else:
        capture_video(args)
