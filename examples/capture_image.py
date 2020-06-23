import os
import logging

from dvgutils import setup_logger, colors
from dvgutils.vis import put_text
from dvgutils.modules import ImageCapture, ShowImage, SaveImage, Metrics, Progress
from dvgutils.pipeline import CaptureImagePipe, Pipeline, ShowImagePipe, SaveImagePipe, MetricsPipe, ProgressPipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="image file or images input path")
    parser.add_argument("-ie", "--image-ext", default="jpg", choices=["jpg", "png"],
                        help="image extension (default: jpg)")
    parser.add_argument("-c", "--contains", type=str,
                        help="image name should contain given string")
    parser.add_argument("-o", "--output", type=str,
                        help="path to output directory")
    parser.add_argument("--no-display", dest='display', action="store_false",
                        help="hide display window")
    parser.add_argument('--progress', action='store_true',
                        help="show progress info")
    parser.add_argument("--pipeline", action="store_true",
                        help="run as pipeline")

    return vars(parser.parse_args())


def capture_image(args):
    logger = logging.getLogger(__name__)

    # Setup processing modules
    image_capture = ImageCapture(args["input"], "." + args["image_ext"], args["contains"])
    save_image = SaveImage(args["output"], args["image_ext"]) if args["output"] else None
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
            if args["input"] != filename:
                # Get rid of input path from filename
                filename = os.path.relpath(filename, start=args["input"])
            else:
                filename = os.path.basename(filename)
            # Getting the name of the file without the extension
            name = os.path.splitext(filename)[0]

            # Visualize image filename
            put_text(image, filename, (2, 2), org_pos="tl",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)

            if save_image:
                save_image(image, name)

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
        filename = data["filename"]

        # Visualize image filename
        put_text(vis_image, filename, (2, 2), org_pos="tl",
                 bg_color=colors.get("white").bgr(), bg_alpha=0.5)

        return data


def capture_image_pipeline(args):
    logger = logging.getLogger(__name__)

    # Setup pipeline steps
    capture_image_pipe = CaptureImagePipe(args["input"], "." + args["image_ext"], args["contains"])
    visualize_data_pipe = VisualizeDataPipe("vis_image")
    save_image_pipe = SaveImagePipe("vis_image", args["output"], args["image_ext"]) if args["output"] else None
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
