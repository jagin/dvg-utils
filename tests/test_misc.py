import numpy as np

from dvgutils.misc import remap, clip_points, str_to_sec


def test_remap():
    assert remap(25.0, 0.0, 100.0, 1.0, -1.0) == 0.5
    assert remap(25.0, 100.0, -100.0, -1.0, 1.0) == -0.25
    assert remap(-125.0, -100.0, -200.0, 1.0, -1.0) == 0.5
    assert remap(-125.0, -200.0, -100.0, -1.0, 1.0) == 0.5
    # even when value is out of bound
    assert remap(-20.0, 0.0, 100.0, 0.0, 1.0) == -0.2


def test_clip_points():
    assert np.array_equal(clip_points([[12, 98]], 100, 100),
                          [[12, 98]])
    assert np.array_equal(clip_points([[-1, 101]], 100, 100),
                          [[0, 100]])
    assert np.array_equal(clip_points([[-1, 101], [23, 45], [0, 121]], 100, 100),
                          [[0, 100], [23, 45], [0, 100]])
    assert np.array_equal(clip_points([[-1, 101], [23, 45], [0, 121]], 100, 120),
                          [[0, 101], [23, 45], [0, 120]])

def test_str_to_sec():
    assert str_to_sec("0:1") == 1
    assert str_to_sec("1:0") == 60
    assert str_to_sec("1:0:0") == 3600
    assert str_to_sec("1:1:1") == 3661
    assert str_to_sec("02:23") == 143
    assert str_to_sec("00:02:23") == 143
    assert str_to_sec("01:02:23") == 3743
    assert str_to_sec("1:0:0.100") == 3600.1
    assert str_to_sec("1:0:0.19") == 3600.019
    assert str_to_sec("1:0:0.765") == 3600.765