import numpy as np


def decode_fourcc(fourcc_code):
    """Decode FOURCC code into string."""
    fourcc_code = int(fourcc_code)

    return "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])


def remap(value, from_min_value, from_max_value, to_min_value, to_max_value):
    """Remap value from from_min_value:from_max_value range to to_min_value:to_max_value range"""
    # Check reversed input range
    reverse_input = False
    from_min = min(from_min_value, from_max_value)
    from_max = max(from_min_value, from_max_value)
    if from_min != from_min_value:
        reverse_input = True

    # Check reversed output range
    reverse_output = False
    to_min = min(to_min_value, to_max_value)
    to_max = max(to_min_value, to_max_value)
    if to_min != to_min_value :
        reverse_output = True

    portion = (value - from_min) * (to_max - to_min) / (from_max - from_min) if not reverse_input \
        else (from_max - value) * (to_max - to_min) / (from_max - from_min)

    new_value = portion + to_min if not reverse_output else to_max - portion

    return new_value


def clip_points(pts, width, height):
    """Clip points coordinates to image width and height.

    :param list[(int,int)] | numpy.ndarray pts: list of points
    :param int width: image width
    :param int height: image height

    :returns: cliped points
    :rtype: numpy.ndarray
    """
    pts = np.array(pts)
    pts[:, :1] = np.clip(pts[:, :1], 0, width)
    pts[:, 1:3] = np.clip(pts[:, 1:3], 0, height)

    return pts


def str_to_sec(time_str):
    """Convert string to seconds.

    :param str time_str: time in string format
    :returns: seconds
    :rtype: float
    """
    time_str = time_str.split(".")
    sec = sum(x * int(t) for x, t in zip([1, 60, 3600], reversed(time_str[0].split(":"))))
    msec = int(time_str[1])/1000 if len(time_str) > 1 else 0.0
    return sec + msec
