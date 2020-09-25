import logging
import time

from dvgutils import setup_logger, load_config, colors
from dvgutils.vis import put_text
from dvgutils.modules import VideoCapture, ShowImage, Metrics, Progress


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--conf", default="config/capture_multi_video.yml",
                        help="Path to the input configuration file (default: config/capture_multi_video.yml)")
    parser.add_argument("-cfo", "--conf-overwrites", nargs="+", type=str)
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="don't display progress")

    return vars(parser.parse_args())


def visualize_frame_info(vis_image, frame_num, fps):
    h, w = vis_image.shape[:2]

    # Visualize frame number
    put_text(vis_image, f"{frame_num}", (w, h), org_pos="br",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)
    # Visualize FPS
    put_text(vis_image, f"{fps:.2f} fps", (0, h), org_pos="bl",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)


def capture_multi_video(args):
    logger = logging.getLogger(__name__)
    conf = load_config(args["conf"], args["conf_overwrites"])

    # Setup processing modules
    video_capture_1 = VideoCapture(conf["videoCapture1"]).open()
    video_capture_2 = VideoCapture(conf["videoCapture2"]).open()

    show_image_1 = ShowImage("Video 1")
    show_image_2 = ShowImage("Video 2")
    metrics = Metrics().start()
    progress = Progress(disable=not args["progress"])

    try:
        logger.info("Capturing...")

        idx = 0
        start_time = time.perf_counter()
        while True:
            # Grab the frame
            frame_1 = video_capture_1.read()
            frame_2 = video_capture_2.read()
            if frame_1 is None or frame_2 is None:
                break
            # Calculate FPS
            fps = (idx + 1) / (time.perf_counter() - start_time)

            vis_frame_1 = frame_1.copy()
            vis_frame_2 = frame_2.copy()

            # Visualize data
            #
            # Metrics
            visualize_frame_info(vis_frame_1, idx, fps)
            visualize_frame_info(vis_frame_2, idx, fps)

            show_1 = show_image_1(vis_frame_1)
            show_2 = show_image_2(vis_frame_2)
            if not show_1 or not show_2:
                break

            metrics.update()
            progress.update()

            idx += 1

        logger.info(f"{len(metrics)} it, "
                    f"{metrics.elapsed():.3f} s, "
                    f"{metrics.sec_per_iter():.3f} s/it, "
                    f"{metrics.iter_per_sec():.2f} it/s")
    except KeyboardInterrupt:
        logger.warning("Got Ctrl+C!")
    finally:
        # Cleanup resources
        progress.close()
        show_image_1.close()
        show_image_2.close()
        video_capture_1.close()
        video_capture_2.close()


if __name__ == "__main__":
    setup_logger()

    args = parse_args()
    capture_multi_video(args)
