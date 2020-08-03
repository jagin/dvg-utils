from dvgutils.vis import resize, rectangle_overlay, put_text
import cv2
import os


class TestVis:
    def test_resize(self):
        image_path = "assets/images/friends/friends_01.jpg"

        output_path = "output/tests/"
        dirname = os.path.abspath(output_path)
        os.makedirs(dirname, exist_ok=True)

        image = cv2.imread(image_path)

        image_resized_w = resize(image, width=300)
        cv2.imwrite(output_path + "image_resized_w.png", image_resized_w)
        assert image_resized_w.shape[:2][1] == 300

        image_resized_h = resize(image, height=300)
        cv2.imwrite(output_path + "image_resized_h.png", image_resized_h)
        assert image_resized_h.shape[:2][0] == 300

        image_resized_h_w = resize(image, width=300, height=300)
        cv2.imwrite(output_path + "image_resized_h_w.png", image_resized_h_w)
        assert image_resized_h_w.shape[:2] == (300, 300)

        image_resized_h_w_p = resize(image, width=300, height=300, pad=True)
        cv2.imwrite(output_path + "image_resized_h_w_p.png", image_resized_h_w_p)
        assert image_resized_h_w_p.shape[:2] == (300, 300)

    def test_rectangle_overlay(self):
        image_path = "assets/images/friends/friends_01.jpg"

        output_path = "output/tests/"
        dirname = os.path.abspath(output_path)
        os.makedirs(dirname, exist_ok=True)

        image = cv2.imread(image_path)

        image_overlay_red_a1 = image.copy()
        rectangle_overlay(image_overlay_red_a1, (200, 200), (500, 500), (0, 0, 255), 1)
        cv2.imwrite(output_path + "image_overlay_red_a1.png", image_overlay_red_a1)
        assert image_overlay_red_a1[300:400, 300:400, 2].sum() > image[300:400, 300:400, 2].sum()

    def test_put_text(self):
        image_path = "assets/images/friends/friends_01.jpg"

        output_path = "output/tests/"
        dirname = os.path.abspath(output_path)
        os.makedirs(dirname, exist_ok=True)

        image = cv2.imread(image_path)
        im_h, im_w = image.shape[:2]

        image_text_redt_tl = image.copy()
        cv2.circle(image_text_redt_tl, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_tl, "TEST TEXT", (500, 500), color=(0, 0, 255), org_pos="tl", font_scale=3)
        image_text_redt_tl = cv2.line(image_text_redt_tl, (500, 500), (500, im_h), (0, 255, 0), 1)
        image_text_redt_tl = cv2.line(image_text_redt_tl, (500, 500), (im_w, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_tl.png", image_text_redt_tl)
        assert image_text_redt_tl[501:600, 501:600, 2].sum() > image[501:600, 501:600, 2].sum()

        image_text_redt_tr = image.copy()
        cv2.circle(image_text_redt_tr, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_tr, "TEST TEXT", (500, 500), color=(0, 0, 255), org_pos="tr", font_scale=3)
        image_text_redt_tr = cv2.line(image_text_redt_tr, (500, 500), (500, im_h), (0, 255, 0), 1)
        image_text_redt_tr = cv2.line(image_text_redt_tr, (500, 500), (0, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_tr.png", image_text_redt_tr)
        assert image_text_redt_tr[501:600, 300:400, 2].sum() > image[501:600, 300:400, 2].sum()

        image_text_redt_bl = image.copy()
        cv2.circle(image_text_redt_bl, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_bl, "TEST TEXT", (500, 500), color=(0, 0, 255), org_pos="bl", font_scale=3)
        image_text_redt_bl = cv2.line(image_text_redt_bl, (500, 500), (500, 0), (0, 255, 0), 1)
        image_text_redt_bl = cv2.line(image_text_redt_bl, (500, 500), (im_w, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_bl.png", image_text_redt_bl)
        assert image_text_redt_bl[400:501, 501:600, 2].sum() > image[400:501, 501:600, 2].sum()

        image_text_redt_br = image.copy()
        cv2.circle(image_text_redt_br, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_br, "TEST TEXT", (500, 500), color=(0, 0, 255), org_pos="br", font_scale=3)
        image_text_redt_br = cv2.line(image_text_redt_br, (500, 500), (500, 0), (0, 255, 0), 1)
        image_text_redt_br = cv2.line(image_text_redt_br, (500, 500), (0, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_br.png", image_text_redt_br)
        assert image_text_redt_br[400:501, 400:501, 2].sum() > image[400:501, 400:501, 2].sum()

        image_text_redt_blueb_tl = image.copy()
        cv2.circle(image_text_redt_blueb_tl, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_tl, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0), bg_alpha=1,
                 org_pos="tl", font_scale=3)
        image_text_redt_blueb_tl = cv2.line(image_text_redt_blueb_tl, (500, 500), (500, im_h), (0, 255, 0), 1)
        image_text_redt_blueb_tl = cv2.line(image_text_redt_blueb_tl, (500, 500), (im_w, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_tl.png", image_text_redt_blueb_tl)
        assert image_text_redt_blueb_tl[501:600, 501:600, 0].sum() > image[501:600, 501:600, 0].sum()

        image_text_redt_blueb_tr = image.copy()
        cv2.circle(image_text_redt_blueb_tr, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_tr, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0), bg_alpha=1,
                 org_pos="tr", font_scale=3)
        image_text_redt_blueb_tr = cv2.line(image_text_redt_blueb_tr, (500, 500), (500, im_h), (0, 255, 0), 1)
        image_text_redt_blueb_tr = cv2.line(image_text_redt_blueb_tr, (500, 500), (0, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_tr.png", image_text_redt_blueb_tr)
        assert image_text_redt_blueb_tr[501:600, 300:400, 0].sum() > image[501:600, 300:400, 0].sum()

        image_text_redt_blueb_bl = image.copy()
        cv2.circle(image_text_redt_blueb_bl, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_bl, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0), bg_alpha=1,
                 org_pos="bl", font_scale=3)
        image_text_redt_blueb_bl = cv2.line(image_text_redt_blueb_bl, (500, 500), (500, 0), (0, 255, 0), 1)
        image_text_redt_blueb_bl = cv2.line(image_text_redt_blueb_bl, (500, 500), (im_w, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_bl.png", image_text_redt_blueb_bl)
        assert image_text_redt_blueb_bl[400:501, 501:600, 0].sum() > image[400:501, 501:600, 0].sum()

        image_text_redt_blueb_br = image.copy()
        cv2.circle(image_text_redt_blueb_br, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_br, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0), bg_alpha=1,
                 org_pos="br", font_scale=3)
        image_text_redt_blueb_br = cv2.line(image_text_redt_blueb_br, (500, 500), (500, 0), (0, 255, 0), 1)
        image_text_redt_blueb_br = cv2.line(image_text_redt_blueb_br, (500, 500), (0, 500), (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_br.png", image_text_redt_blueb_br)
        assert image_text_redt_blueb_br[400:501, 400:501, 0].sum() > image[400:501, 400:501, 0].sum()

        image_text_redt_blueb_tl_pad = image.copy()
        cv2.circle(image_text_redt_blueb_tl_pad, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_tl_pad, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0),
                 bg_alpha=1,
                 org_pos="tl", font_scale=3, padding=20)
        image_text_redt_blueb_tl_pad = cv2.line(image_text_redt_blueb_tl_pad, (500, 500), (500, im_h), (0, 255, 0), 1)
        image_text_redt_blueb_tl_pad = cv2.line(image_text_redt_blueb_tl_pad, (500, 500), (im_w, 500), (0, 255, 0), 1)
        image_text_redt_blueb_tl_pad = cv2.line(image_text_redt_blueb_tl_pad, (500 + 20, 500 + 20), (500 + 20, im_h),
                                                (0, 255, 0), 1)
        image_text_redt_blueb_tl_pad = cv2.line(image_text_redt_blueb_tl_pad, (500 + 20, 500 + 20), (im_w, 500 + 20),
                                                (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_tl_pad.png", image_text_redt_blueb_tl_pad)

        image_text_redt_blueb_tr_pad = image.copy()
        cv2.circle(image_text_redt_blueb_tr_pad, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_tr_pad, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0),
                 bg_alpha=1,
                 org_pos="tr", font_scale=3, padding=20)
        image_text_redt_blueb_tr_pad = cv2.line(image_text_redt_blueb_tr_pad, (500, 500), (500, im_h), (0, 255, 0), 1)
        image_text_redt_blueb_tr_pad = cv2.line(image_text_redt_blueb_tr_pad, (500, 500), (0, 500), (0, 255, 0), 1)
        image_text_redt_blueb_tr_pad = cv2.line(image_text_redt_blueb_tr_pad, (500 - 20, 500 + 20), (500 - 20, im_h),
                                                (0, 255, 0), 1)
        image_text_redt_blueb_tr_pad = cv2.line(image_text_redt_blueb_tr_pad, (500 - 20, 500 + 20), (0, 500 + 20),
                                                (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_tr_pad.png", image_text_redt_blueb_tr_pad)

        image_text_redt_blueb_bl_pad = image.copy()
        cv2.circle(image_text_redt_blueb_bl_pad, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_bl_pad, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0),
                 bg_alpha=1,
                 org_pos="bl", font_scale=3, padding=20)
        image_text_redt_blueb_bl_pad = cv2.line(image_text_redt_blueb_bl_pad, (500, 500), (500, 0), (0, 255, 0), 1)
        image_text_redt_blueb_bl_pad = cv2.line(image_text_redt_blueb_bl_pad, (500, 500), (im_w, 500), (0, 255, 0), 1)
        image_text_redt_blueb_bl_pad = cv2.line(image_text_redt_blueb_bl_pad, (500 + 20, 500 - 20), (500 + 20, 0),
                                                (0, 255, 0), 1)
        image_text_redt_blueb_bl_pad = cv2.line(image_text_redt_blueb_bl_pad, (500 + 20, 500 - 20), (im_w, 500 - 20),
                                                (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_bl_pad.png", image_text_redt_blueb_bl_pad)

        image_text_redt_blueb_br_pad = image.copy()
        cv2.circle(image_text_redt_blueb_br_pad, (500, 500), 2, (0, 255, 0))
        put_text(image_text_redt_blueb_br_pad, "TEST TEXT", (500, 500), color=(0, 0, 255), bg_color=(255, 0, 0),
                 bg_alpha=1,
                 org_pos="br", font_scale=3, padding=20)
        image_text_redt_blueb_br_pad = cv2.line(image_text_redt_blueb_br_pad, (500, 500), (500, 0), (0, 255, 0), 1)
        image_text_redt_blueb_br_pad = cv2.line(image_text_redt_blueb_br_pad, (500, 500), (0, 500), (0, 255, 0), 1)
        image_text_redt_blueb_br_pad = cv2.line(image_text_redt_blueb_br_pad, (500 - 20, 500 - 20), (500 - 20, 0),
                                                (0, 255, 0), 1)
        image_text_redt_blueb_br_pad = cv2.line(image_text_redt_blueb_br_pad, (500 - 20, 500 - 20), (0, 500 - 20),
                                                (0, 255, 0), 1)
        cv2.imwrite(output_path + "image_text_redt_blueb_br_pad.png", image_text_redt_blueb_br_pad)
