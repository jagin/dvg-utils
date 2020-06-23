import logging
import os

import cv2

from dvgutils import setup_logger, load_config, colors
from dvgutils.vis import put_text, rectangle_overlay
from dvgutils.modules import ImageCapture, ShowImage, SaveImage, Metrics, Progress
from dvgutils.pipeline import CaptureImagePipe, Pipeline, ShowImagePipe, SaveImagePipe, MetricsPipe, ProgressPipe

from modules.face_detector import FaceDetector
from pipeline.detect_face_pipe import DetectFacePipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/config.yml",
                        help="Path to the input configuration file (default: config/config.yml)")
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
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="don't display progress")
    parser.add_argument("--pipeline", action="store_true",
                        help="run as pipeline")

    return vars(parser.parse_args())


def detect_face(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup processing modules
    image_capture = ImageCapture(args["input"], "." + args["image_ext"], args["contains"])
    face_detector = FaceDetector(conf["faceDetector"])
    save_image = SaveImage(args["output"], args["image_ext"]) if args["output"] else None
    show_image = ShowImage("Image", delay=0) if args["display"] else None
    metrics = Metrics().start()
    progress = Progress(disable=not args["progress"])

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

            # Detect faces
            face_locations = face_detector.detect(image)

            # Visualize data
            #
            # Image filename
            put_text(image, filename, (2, 2), org_pos="tl",
                     bg_color=colors.get("white").bgr(), bg_alpha=0.5)
            # Face bounding boxes
            for face_location in face_locations:
                (start_x, start_y, end_x, end_y) = face_location[0:4]
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
                rectangle_overlay(image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)

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
        # Clean up resources
        progress.close()
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
        face_locations = data["face_locations"]

        # Image filename
        put_text(vis_image, filename, (2, 2), org_pos="tl",
                 bg_color=colors.get("white").bgr(), bg_alpha=0.5)
        # Face bounding boxes
        for face_location in face_locations:
            (start_x, start_y, end_x, end_y) = face_location[0:4]
            cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
            rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)

        return data


def detect_face_pipeline(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"])

    # Setup pipeline steps
    capture_image_pipe = CaptureImagePipe(args["input"], "." + args["image_ext"], args["contains"])
    detect_face_pipe = DetectFacePipe(conf["faceDetector"])
    visualize_data_pipe = VisualizeDataPipe("vis_image")
    save_image_pipe = SaveImagePipe("vis_image", args["output"], args["image_ext"]) if args["output"] else None
    show_image_pipe = ShowImagePipe("vis_image", "Video", delay=0) if args["display"] else None
    metrics_pipe = MetricsPipe()
    progress_pipe = ProgressPipe(disable=not args["progress"])

    # Create pipeline
    pipeline = Pipeline(capture_image_pipe)
    pipeline.map(detect_face_pipe)
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
        detect_face_pipeline(args)
    else:
        detect_face(args)
