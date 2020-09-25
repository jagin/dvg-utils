from ..modules import Progress


class ProgressPipe:
    def __init__(self, **kwargs):
        self.progress = Progress(**kwargs)

    def __call__(self, data):
        return self.update(data)

    def update(self, data):
        self.progress.update()

        return data

    def close(self):
        self.progress.close()
