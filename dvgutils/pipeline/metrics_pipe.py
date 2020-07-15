from ..modules import Metrics


class MetricsPipe:
    def __init__(self):
        self.metrics = Metrics().start()

    def __call__(self, data):
        return self.update(data)

    def update(self, data):
        self.metrics.update()

        return data
