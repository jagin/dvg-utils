import cv2

from dvgutils.vis import resize


class MotionDetector:

    def __init__(self, min_area=500, delta_thresh=5):
        self.min_area = min_area
        self.delta_thresh = delta_thresh
        self.avg_frame = None
        self.scale_h = None
        self.scale_w = None

    def preprocess(self, image):
        # Resize the frame, convert it to grayscale, and blur it
        image = resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        return gray

    def get_scale(self, img_dest, img_scaled):
        h_dest, w_dest = img_dest.shape[:2]
        h_scaled, w_scaled = img_scaled.shape[:2]

        return h_dest / h_scaled, w_dest / w_scaled

    def detect(self, frame):
        locations = []
        gray = self.preprocess(frame)

        # If the first frame is None, initialize it
        if self.avg_frame is None:
            self.avg_frame = gray.copy().astype("float")
            self.scale_h, self.scale_w = self.get_scale(frame, gray)
            return locations

        # Accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, self.avg_frame, 0.5)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))

        # Threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frame_delta, self.delta_thresh, 255,
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

        if cnts is not None:
            # Loop over the contours
            for c in cnts:
                # If the contour is too small, ignore it
                if cv2.contourArea(c) < self.min_area:
                    continue

                # Compute the bounding box for the contour and append the location
                (x, y, w, h) = cv2.boundingRect(c)
                locations.append((round(self.scale_w * x), round(self.scale_h * y),
                                  round(self.scale_w * (x + w)), round(self.scale_h * (y + h))))

        return locations
