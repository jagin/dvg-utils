import logging
import os

from dvgutils import setup_logger, load_config
from dvgutils.modules import ImageCapture, ShowImage, SaveImage, Metrics, Progress
from dvgutils.pipeline import CaptureImagePipe, Pipeline, ShowImagePipe, SaveImagePipe, MetricsPipe, ProgressPipe

from utils.vis import visualize_image_info, visualize_face_locations
from modules.face_detector import FaceDetector
from pipeline.detect_face_pipe import DetectFacePipe


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/detect_face_image.yml",
                        help="Path to the input configuration file (default: config/detect_face_image.yml)")
    parser.add_argument("-cfo", "--conf-overwrites", nargs="+", type=str)
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
    conf = load_config(args["conf"], args["conf_overwrites"])

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
            # Visualize image filename
            visualize_image_info(image, filename)
            # Visualize face locations
            visualize_face_locations(image, face_locations)

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

        self.visualize_image_info(data)
        self.visualize_face_locations(data)

        return data

    def visualize_image_info(self, data):
        vis_image = data[self.image_key]
        filename = data["filename"]

        visualize_image_info(vis_image, filename)

    def visualize_face_locations(self, data):
        vis_image = data[self.image_key]
        face_locations = data["face_locations"]

        visualize_face_locations(vis_image, face_locations)

def detect_face_pipeline(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"], args["conf_overwrites"])

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
