import os
import logging

from dvgutils import setup_logger, load_config
from dvgutils.modules import ImageCapture, ShowImage, SaveImage, Metrics, Progress
from dvgutils.pipeline import CaptureImagePipe, Pipeline, ShowImagePipe, SaveImagePipe, MetricsPipe, ProgressPipe

from utils.vis import visualize_image_info


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/capture_image.yml",
                        help="Path to the input configuration file (default: config/capture_image.yml)")
    parser.add_argument("-cfo", "--conf-overwrites", nargs="+", type=str)
    parser.add_argument("-o", "--output", type=str,
                        help="path to output directory")
    parser.add_argument("--no-display", dest='display', action="store_false",
                        help="hide display window")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="don't display progress")
    parser.add_argument("--pipeline", action="store_true",
                        help="run as pipeline")

    return vars(parser.parse_args())


def capture_image(args):
    # Setup logger
    logger = logging.getLogger(__name__)
    # Read configuration
    conf = load_config(args["conf"], args["conf_overwrites"])

    # Setup processing modules
    image_capture = ImageCapture(conf["imageCapture"])
    save_image = SaveImage(args["output"]) if args["output"] else None
    show_image = ShowImage("Image", delay=0) if args["display"] else None
    metrics = Metrics().start()
    progress = Progress()

    try:
        logger.info("Capturing...")
        while True:
            # Grab the frame
            filename, image = image_capture.read()
            if image is None:
                break

            # Get rid of input path from filename
            if conf["imageCapture"]["path"] != filename:
                # Get rid of input path from filename
                filename = os.path.relpath(filename, start=conf["imageCapture"]["path"])
            else:
                filename = os.path.basename(filename)

            # Visualize data
            #
            # Visualize image filename
            visualize_image_info(image, filename)

            if save_image:
                save_image(image, filename)

            if show_image:
                show = show_image(image)
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
        progress.close()
        # Clean up resources
        if show_image:
            show_image.close()


class VisualizeDataPipe:
    def __init__(self, image_key="vis_image"):
        self.image_key = image_key

    def __call__(self, data):
        return self.visualize(data)

    def visualize(self, data):
        vis_image = data["image"].copy()
        data[self.image_key] = vis_image

        self.visualize_image_info(data)

        return data

    def visualize_image_info(self, data):
        vis_image = data[self.image_key]
        filename = data["filename"]

        visualize_image_info(vis_image, filename)


def capture_image_pipeline(args):
    # Setup logger
    logger = logging.getLogger(__name__)
    # Read configuration
    conf = load_config(args["conf"], args["conf_overwrites"])

    # Setup pipeline steps
    capture_image_pipe = CaptureImagePipe(conf["imageCapture"])
    visualize_data_pipe = VisualizeDataPipe("vis_image")
    save_image_pipe = SaveImagePipe("vis_image", args["output"]) if args["output"] else None
    show_image_pipe = ShowImagePipe("vis_image", "Video", delay=0) if args["display"] else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe()

    # Create pipeline
    pipeline = Pipeline(capture_image_pipe)
    pipeline.map(visualize_data_pipe)
    pipeline.map(save_image_pipe)
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
        capture_image_pipeline(args)
    else:
        capture_image(args)
