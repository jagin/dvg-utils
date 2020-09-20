import cv2

from dvgutils import colors
from dvgutils.vis import put_text, rectangle_overlay


def visualize_image_info(vis_image, filename):
    put_text(vis_image, filename, (0, 0), org_pos="tl",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)


def visualize_frame_info(vis_image, frame_num, fps):
    h, w = vis_image.shape[:2]
    # Visualize frame number
    put_text(vis_image, f"{frame_num}", (w, h), org_pos="br",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)
    # Visualize FPS
    put_text(vis_image, f"{fps:.2f} fps", (0, h), org_pos="bl",
             bg_color=colors.get("white").bgr(), bg_alpha=0.5)


def visualize_face_locations(vis_image, face_locations):
    for face_location in face_locations:
        (start_x, start_y, end_x, end_y) = face_location[0:4]
        cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
        rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)


def visualize_motion_locations(vis_image, motion_locations):
    if motion_locations:
        for motion_location in motion_locations:
            (x1, y1, x2, y2) = motion_location
            rectangle_overlay(vis_image, (x1, y1), (x2, y2), colors.get("red").bgr(), 0.5)
        put_text(vis_image, "MOTION DETECTED!", (0, 0), org_pos="tl",
                 bg_color=colors.get("red").bgr())


def visualize_object_locations(vis_image, object_locations):
    if object_locations:
        for object_location in object_locations:
            (start_x, start_y, end_x, end_y, label, confidence) = object_location
            cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
            rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)
            put_text(vis_image, f"{label} {confidence:.2f}", (start_x - 1, start_y - 1), org_pos="bl",
                     bg_color=colors.get("green").bgr())


def visualize_object_counter(vis_image, crossed_in_out, line):
    (crossed_in, crossed_out) = crossed_in_out

    cv2.line(vis_image, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 2)

    # Construct a tuple of information we will be displaying on the
    # Frame
    info = [
        ("Crossed In", crossed_in),
        ("Crossed Out", crossed_out),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        put_text(vis_image, text, (0, ((i * 22) + 20)), org_pos="bl",
                 bg_color=colors.get("white").bgr(), bg_alpha=0.5)


def visualize_tracked_object_locations(vis_image, tracked_objects):
    # Loop over the tracked objects
    for tracked_object in tracked_objects:
        object_id = tracked_object["object_id"]
        centroids = tracked_object["centroids"]
        bbox_dims = tracked_object["bbox_dims"]
        (x_coord, y_coord) = centroids[-1]

        # Draw bounding box of the object using the last known bounding box dimensions from detection
        (bbox_w, bbox_h) = bbox_dims / 2
        (start_x, end_x, start_y, end_y) = int(round(x_coord - bbox_w)), int(round(x_coord + bbox_w)), \
                                           int(round(y_coord - bbox_h)), int(round(y_coord + bbox_h))
        cv2.rectangle(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 2)
        rectangle_overlay(vis_image, (start_x, start_y), (end_x, end_y), colors.get("green").bgr(), 0.5)

        # Draw both the ID of the object and the centroid of the object on the output frame
        put_text(vis_image, f"Id {object_id}", (start_x - 1, start_y - 1), org_pos="bl",
                 bg_color=colors.get("green").bgr())
        cv2.circle(vis_image, (x_coord, y_coord), 4, colors.get("green").bgr(), -1)

        # For each object draw the line which tells us its position in the last 30 frames
        if len(centroids) >= 31:
            centroids = centroids[-31:-1]
        for x_coord_next, y_coord_next in reversed(centroids):
            cv2.line(vis_image, (x_coord_next, y_coord_next), (x_coord, y_coord), colors.get("red").bgr(), 2)
            x_coord = x_coord_next
            y_coord = y_coord_next