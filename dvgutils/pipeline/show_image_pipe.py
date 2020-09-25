from ..modules import ShowImage
from .observable import observable


class ShowImagePipe:
    def __init__(self, image_key, *args, **kwargs):
        self.image_key = image_key
        self.show_image = ShowImage(*args, **kwargs)

    def __call__(self, data):
        return self.show(data)

    def show(self, data):
        image = data[self.image_key]

        # Show the output frame
        show = self.show_image(image)
        if not show:
            observable.notify("stop")

        return data

    def close(self):
        self.show_image.close()
