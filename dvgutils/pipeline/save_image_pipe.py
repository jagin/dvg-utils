from ..modules.save_image import SaveImage


class SaveImagePipe:
    def __init__(self, image_key, *args, **kwargs):
        self.image_key = image_key
        self.save_video = SaveImage(*args, **kwargs)

    def __call__(self, data):
        return self.save(data)

    def save(self, data):
        image = data[self.image_key]
        name = data["name"]

        self.save_video(image, name)

        return data
