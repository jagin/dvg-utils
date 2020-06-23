import logging

from dvgutils.modules import ImageCapture, ShowImage, SaveVideo, Metrics, Progress


def images_to_video(args):
    logger = logging.getLogger(__name__)

    # Setup processing modules
    image_capture = ImageCapture(args["input"], valid_exts=f".{args['image_ext']}")
    save_video = SaveVideo(args["output"], fps=args["fps"], fourcc=args["codec"])
    show_image = ShowImage("Video") if args["display"] else None
    metrics = Metrics().start()
    progress = Progress(disable=not args["progress"])

    try:
        logger.info("Processing...")
        while True:
            # Grab the frame
            _, image = image_capture.read()
            if image is None:
                break

            save_video(image)

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
        logger.warning("Got Ctrl+C")
    finally:
        # Clean up resources
        progress.close()
        save_video.close()
