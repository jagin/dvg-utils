import cv2
import os

from dvgutils.vis import resize, rectangle_overlay, put_text
from dvgutils import colors

import tests.config as config


class TestVis:
    def test_resize(self):
        output_path = os.path.join(config.OUTPUT_DIR, "tests")
        os.makedirs(output_path, exist_ok=True)

        image_path = os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_01.jpg")
        image = cv2.imread(image_path)

        image_test = resize(image, width=300)
        cv2.imwrite(os.path.join(output_path, "test_resize_w_300.png"), image_test)
        assert image_test.shape[:2][1] == 300

        image_test = resize(image, height=300)
        cv2.imwrite(os.path.join(output_path, "test_resize_h_300.png"), image_test)
        assert image_test.shape[:2][0] == 300

        image_test = resize(image, width=300, height=300)
        cv2.imwrite(os.path.join(output_path, "test_resize_w_300_h_300.png"), image_test)
        assert image_test.shape[:2] == (300, 300)

    def test_resize_horizontal_with_padding(self):
        output_path = os.path.join(config.OUTPUT_DIR, "tests")
        os.makedirs(output_path, exist_ok=True)

        # Horizontal image
        image_path = os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_01.jpg")
        image = cv2.imread(image_path)

        image_test = resize(image, width=300, height=600, pad=True)
        cv2.imwrite(os.path.join(output_path, "test_resize_horizontal_with_padding_w_300_h_600.png"), image_test)
        assert image_test.shape[:2] == (600, 300)

        image_test = resize(image, width=600, height=300, pad=True)
        cv2.imwrite(os.path.join(output_path, "test_resize_horizontal_with_padding_w_600_h_300.png"), image_test)
        assert image_test.shape[:2] == (300, 600)

    def test_resize_vertical_with_padding(self):
        output_path = os.path.join(config.OUTPUT_DIR, "tests")
        os.makedirs(output_path, exist_ok=True)

        # Vertical image
        image_path = os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_05.jpg")
        image = cv2.imread(image_path)

        image_test = resize(image, width=300, height=600, pad=True)
        cv2.imwrite(os.path.join(output_path, "test_resize_vertical_with_padding_w_300_h_600_pad.png"), image_test)
        assert image_test.shape[:2] == (600, 300)

        image_test = resize(image, width=600, height=300, pad=True)
        cv2.imwrite(os.path.join(output_path, "test_resize_vertical_with_padding_w_600_h_300_pad.png"), image_test)
        assert image_test.shape[:2] == (300, 600)

    def test_rectangle_overlay(self):
        output_path = os.path.join(config.OUTPUT_DIR, "tests")
        os.makedirs(output_path, exist_ok=True)

        image_path = os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_01.jpg")
        image = cv2.imread(image_path)

        image_overlay_red_a1 = image.copy()
        rectangle_overlay(image_overlay_red_a1, (200, 200), (500, 500), colors.get("red").bgr(), 0.5)
        cv2.imwrite(os.path.join(output_path, "test_rectangle_overlay.png"), image_overlay_red_a1)
        assert image_overlay_red_a1[200:500, 200:500, 2].sum() > image[200:500, 200:500, 2].sum()

    def test_put_text(self):
        output_path = os.path.join(config.OUTPUT_DIR, "tests")
        os.makedirs(output_path, exist_ok=True)

        image_path = os.path.join(config.ASSETS_IMAGES_DIR, "friends", "friends_01.jpg")
        image = cv2.imread(image_path)
        test_text = "TEST TEXT"
        text_org = (500, 500)
        text_padding = 20

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), org_pos="tl", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_tl.png"), image_test)
        assert text_org == pt1

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), org_pos="tr", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_tr.png"), image_test)
        assert text_org == (pt2[0], pt1[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), org_pos="bl", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_bl.png"), image_test)
        assert text_org == (pt1[0], pt2[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), org_pos="br", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_br.png"), image_test)
        assert text_org == (pt2[0], pt2[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="tl", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_tl.png"), image_test)
        assert text_org == (pt1[0], pt1[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="tr", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_tr.png"), image_test)
        assert text_org == (pt2[0], pt1[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="bl", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_bl.png"), image_test)
        assert text_org == (pt1[0], pt2[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="br", font_scale=3)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_br.png"), image_test)
        assert text_org == (pt2[0], pt2[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="tl", font_scale=3, padding=text_padding)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.rectangle(image_test, (pt1[0] + text_padding, pt1[1] + text_padding),
                      (pt2[0] - text_padding, pt2[1] - text_padding), colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_tl_pad.png"), image_test)
        assert text_org == (pt1[0], pt1[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="tr", font_scale=3, padding=text_padding)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.rectangle(image_test, (pt1[0] + text_padding, pt1[1] + text_padding),
                      (pt2[0] - text_padding, pt2[1] - text_padding), colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_tr_pad.png"), image_test)
        assert text_org == (pt2[0], pt1[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="bl", font_scale=3, padding=text_padding)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.rectangle(image_test, (pt1[0] + text_padding, pt1[1] + text_padding),
                      (pt2[0] - text_padding, pt2[1] - text_padding), colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_bl_pad.png"), image_test)
        assert text_org == (pt1[0], pt2[1])

        image_test = image.copy()
        cv2.circle(image_test, text_org, 4, colors.get("red").bgr())
        pt1, pt2 = put_text(image_test, test_text, text_org,
                            color=colors.get("red").bgr(), bg_color=colors.get("blue").bgr(), bg_alpha=0.6,
                            org_pos="br", font_scale=3, padding=text_padding)
        cv2.rectangle(image_test, pt1, pt2, colors.get("green").bgr(), 1)
        cv2.rectangle(image_test, (pt1[0] + text_padding, pt1[1] + text_padding),
                      (pt2[0] - text_padding, pt2[1] - text_padding), colors.get("green").bgr(), 1)
        cv2.imwrite(os.path.join(output_path, "test_put_text_red_blue_br_pad.png"), image_test)
        assert text_org == (pt2[0], pt2[1])
