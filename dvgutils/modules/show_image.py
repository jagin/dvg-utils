import cv2


class ShowImage:
    def __init__(self, window_name, org=None, exit_key=27, delay=1, flags=cv2.WINDOW_AUTOSIZE):
        self.window_name = window_name
        self.exit_key = exit_key
        self.delay = delay

        self.window_closed = None
        self._wnd_prop_visible = None
        self._wnd_prop_autosize = None

        cv2.namedWindow(self.window_name, flags)
        if org:
            # Set the window position
            x, y = org
            cv2.moveWindow(self.window_name, x, y)

    def __call__(self, image):
        return self.show(image)

    def show(self, image):
        # We are checking both window properties if the window is closed as they
        # behave differently on different platforms (for ex. Ubuntu vs Raspberry Pi)
        wnd_prop_visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        wnd_prop_autosize = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE)

        if self._wnd_prop_visible is not None or self._wnd_prop_autosize is not None:
            self.window_closed = self._wnd_prop_visible != wnd_prop_visible or \
                                 self._wnd_prop_autosize != wnd_prop_autosize
        else:
            self.window_closed = False

        self._wnd_prop_visible = wnd_prop_visible
        self._wnd_prop_autosize = wnd_prop_autosize

        if not self.window_closed:
            cv2.imshow(self.window_name, image)

            # Esc key pressed or window closed? Still show window?
            if self.delay > 0:
                key = cv2.waitKey(self.delay)
                show = key != self.exit_key
            else:
                # cv2.waitKey() creates a new thread.  If you press the “x” (closing icon) in GUI the main thread
                # will be waiting for the cv2.waitKey() to be executed. Since the window is closed,
                # the cv2.waitKey() has no chance to be executed - no window, no key press, deadlock.
                # Here's the trick:
                while True:
                    wnd_prop_visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                    wnd_prop_autosize = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE)
                    key = cv2.waitKey(100)
                    self.window_closed = self._wnd_prop_visible != wnd_prop_visible or \
                                         self._wnd_prop_autosize != wnd_prop_autosize
                    show = key != self.exit_key and not self.window_closed
                    if key > -1 or self.window_closed:
                        break
        else:
            show = False

        return show

    def close(self):
        if not self.window_closed:
            cv2.destroyWindow(self.window_name)
            self.window_closed = True
