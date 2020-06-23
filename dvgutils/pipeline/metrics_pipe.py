from ..modules import Metrics
from .observable import observable

class MetricsPipe:
    def __init__(self):
        self.metrics = Metrics().start()

    def __call__(self, data):
        return self.update(data)

    def update(self, data):
        self.metrics.update()

        observable.notify("metrics", {
            "iteration": len(self.metrics),
            "elapsed": self.metrics.elapsed(),
            "iter_per_sec": self.metrics.iter_per_sec(),
            "sec_per_iter": self.metrics.sec_per_iter()
        })

        return data
