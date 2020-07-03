class Pipeline:
    def __init__(self, iterable):
        self.pipes = []
        self.pipeline = iterable
        self.pipes.append(iterable)

    def __iter__(self):
        return self.pipeline

    def map(self, pipe):
        if pipe:
            self.pipeline = map(pipe, self.pipeline)
            self.pipes.append(pipe)

        return self

    def filter(self, pipe):
        if pipe:
            self.pipeline = filter(pipe, self.pipeline)
            self.pipes.append(pipe)

        return self

    def iter(self, pipe):
        if pipe:
            self.pipeline = pipe(self.pipeline)
            self.pipes.append(pipe)

        return self

    def run(self):
        for _ in self.pipeline:
            pass

    def close(self):
        for pipe in reversed(self.pipes):
            if "close" in dir(pipe):
                pipe.close()
