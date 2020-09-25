"""Visual helpers utilizing and extending OpenCV library"""

import cv2
import numpy as np

from .misc import clip_points


def resize(image, width=None, height=None, interp=None, pad=False, pad_color=0):
    """Resize the image down to or up to the specified size.

    Specify width or height for the image size if you want to preserve the aspect ratio of the image.
    You can pad your image with additional border to preserve the aspect ration if needed for custom
    width and height.


    :param numpy.ndarray image: input image
    :param int width: output image width
    :param int height: output image height
    :param int interp: interpolation used to resize the image
    :param bool pad: pad image with borders to preserve aspect ration
    :param (int, int, int) pad_color: pad borders color

    :returns: output image
    :rtype: numpy.ndarray
    """
    src_h, src_w = image.shape[:2]

    if width is None and height is None:
        # No size specified - return original image
        return image

    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        ratio = height / float(src_h)
        width = int(src_w * ratio)

    if height is None:
        # Calculate the ratio of the width and construct the dimensions
        ratio = width / float(src_w)
        height = int(src_h * ratio)

    if width == src_w and height == src_h:
        # There is no change in size - return original image
        return image

    # Select best interpolation method if not specified
    if interp is None:
        if src_h > height or src_w > width:  # Shrinking image
            interp = cv2.INTER_AREA
        else:  # Stretching image
            interp = cv2.INTER_CUBIC

    src_aspect = src_w / src_h  # Aspect ratio of the source image
    aspect = width / height  # Aspect ratio of the new image

    # Compute scaling
    if src_aspect > 1 and aspect < 1 or src_aspect < 1 and aspect < 1:
        new_w = width
        new_h = np.round(new_w / src_aspect).astype(int) if pad else height
    elif src_aspect > 1 and aspect > 1 or src_aspect < 1 and aspect > 1:
        new_h = height
        new_w = np.round(new_h * src_aspect).astype(int) if pad else width
    else:  # Square image
        new_h, new_w = height, width

    # Resize image
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    if pad:
        if len(image.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
            # Color image - set pad color as RGB
            pad_color = [pad_color] * 3

        if aspect > 1:
            pad_horz = (width - new_w) / 2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        elif aspect < 1:
            pad_vert = (height - new_h) / 2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        else:
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # Pad the image with borders
        image = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right,
                                   borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return image


def rectangle_overlay(image, pt1, pt2, color, alpha):
    """Renders the rectangular overlay on the image.

    :param numpy.ndarray image: input image
    :param (int, int) pt1: bottom-left corner of the rectangle
    :param (int, int) pt2: top-right corner of the rectangle
    :param (int, int, int) color: rectangle color
    :param float alpha: alpha for overlay transparency
    """
    h, w = image.shape[:2]
    pt1, pt2 = clip_points([pt1, pt2], w, h)

    roi = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    rect = np.zeros(roi.shape, dtype=np.uint8)
    rect[::] = color
    cv2.addWeighted(rect, alpha, roi, 1 - alpha, 0, roi)


def put_text(image, text, org, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
             color=(0, 0, 0), bg_color=None, bg_alpha=1, thickness=1, line_type=cv2.LINE_AA,
             org_pos="tl", padding=2):
    """Renders the specified text string in the image.

    :param numpy.ndarray image: input image
    :param str text: text string to be drawn
    :param (int, int) org: corner of the text string in the image
        (the position of the corner is determined by org_pos parameter)
    :param int font_face: font type
    :param float font_scale: font scale factor
    :param (int, int, int) color: text color
    :param (int, int, int) bg_color: text background color
    :param float bg_alpha: text background alpha
    :param int thickness: thickness of the lines used to draw a text
    :param int line_type: line type
    :param str org_pos: corner position (org):
        'tl' - top-left, 'tr' - top-right, 'bl' - bottom-left, 'br' - bottom-right
    :param int padding: text padding
    """
    x, y = org
    ret, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

    # Calculate text and background box coordinates
    if org_pos == "tl":  # top-left origin
        bg_rect_pt1 = (x, y)
        bg_rect_pt2 = (x + ret[0] + 2 * padding, y + ret[1] + baseline + 2 * padding)
        text_org = (x + padding, y + ret[1] + padding)
    elif org_pos == "tr":  # top-right origin
        bg_rect_pt1 = (x - ret[0] - 2 * padding, y)
        bg_rect_pt2 = (x, y + ret[1] + baseline + 2 * padding)
        text_org = (x - ret[0] - padding, y + ret[1] + padding)
    elif org_pos == "bl":  # bottom-left origin
        bg_rect_pt1 = (x, y - ret[1] - baseline - 2 * padding)
        bg_rect_pt2 = (x + ret[0] + 2 * padding, y)
        text_org = (x + padding, y - padding - baseline)
    elif org_pos == "br":  # bottom-right origin
        bg_rect_pt1 = (x - ret[0] - 2 * padding, y - ret[1] - baseline - 2 * padding)
        bg_rect_pt2 = (x, y)
        text_org = (x - ret[0] - padding, y - baseline - padding)

    if bg_color:
        # Draw background box
        rectangle_overlay(image, bg_rect_pt1, bg_rect_pt2, bg_color, bg_alpha)

    cv2.putText(image,
                text=text,
                org=text_org,
                fontFace=font_face,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=line_type)

    return bg_rect_pt1, bg_rect_pt2
