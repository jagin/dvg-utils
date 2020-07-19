import logging

from dvgutils import load_config
from dvgutils.modules import VideoCapture, ShowImage, SaveImage, Metrics, Progress
from dvgutils.modules.video_capture import StreamVideoCaptureThreaded


def video_to_images(args):
    logger = logging.getLogger(__name__)

    # Setup processing modules
    if args["input"]:
        input_src = args["input"]
        if input_src.isdigit():
            input_src = int(input_src)
        video_capture = StreamVideoCaptureThreaded(input_src).open()
    else:
        conf = load_config(args["conf"])
        video_capture = VideoCapture(conf["videoCapture"]).open()
    save_image = SaveImage(args["output"], image_ext=args["image_ext"])
    show_image = ShowImage("Video") if args["display"] else None
    metrics = Metrics().start()
    progress = Progress(disable=not args["progress"])

    try:
        logger.info("Processing...")
        frame_idx = 0
        while True:
            # Grab the frame
            frame = video_capture.read()
            if frame is None:
                break

            save_image(frame, f"{frame_idx:06d}")
            frame_idx += 1

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
        logger.warning("Got Ctrl+C")
    except FileExistsError as e:
        logger.error(f"{e}")
    finally:
        # Clean up resources
        progress.close()
        video_capture.close()
