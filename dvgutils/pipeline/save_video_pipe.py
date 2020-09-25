from ..modules.save_video import SaveVideo


class SaveVideoPipe:
    def __init__(self, image_key, *args, **kwargs):
        self.image_key = image_key
        self.save_video = SaveVideo(*args, **kwargs)

    def __call__(self, data):
        return self.save(data)

    def save(self, data):
        image = data[self.image_key]

        self.save_video(image)

        return data

    def close(self):
        self.save_video.close()
