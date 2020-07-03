from collections import deque

from dvgutils.pipeline import Pipeline


class GenerateNumbersPipe:
    def __init__(self, *args):
        self.args = args

    def __iter__(self):
        return self.generator()

    def generator(self):
        for value in range(*self.args):
            data = {
                "value": value
            }
            yield data


class IsMultipleOfPipe:
    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, data):
        return self.is_multiple_of(data)

    def is_multiple_of(self, data):
        return data["value"] % self.factor == 0


class MovingAveragePipe:
    def __init__(self, period=5):
        self.period = period
        self.buff = deque(maxlen=period)

    def __call__(self, data):
        return self.moving_average(data)

    def moving_average(self, data):
        self.buff.append(data["value"])

        if len(self.buff) >= self.period:
            data["avg"] = sum(self.buff) / len(self.buff)

        return data


class BatchPipe:
    def __init__(self, batch_size=5):
        self.batch_size = batch_size

    def __call__(self, iterable):
        return self.generator(iterable)

    def generator(self, iterable):
        batch = []
        batch_no = 1
        for data in iterable:
            batch.append(data)
            if len(batch) == self.batch_size:
                for batch_data in batch:
                    batch_data["batch_no"] = batch_no
                    yield batch_data

                batch.clear()
                batch_no += 1


class PrintPipe:
    def __init__(self, key="value"):
        self.key = key

    def __call__(self, data):
        return self.print(data)

    def print(self, data):
        if self.key in data:
            print(data[self.key])

        return data


class TestPipeline:
    def test_print_pipeline(self, capsys):
        generate_numbers_pipe = GenerateNumbersPipe(10)
        print_pipe = PrintPipe()

        # Create pipeline
        pipeline = Pipeline(generate_numbers_pipe)
        pipeline.map(print_pipe)

        pipeline.run()

        captured = capsys.readouterr()

        assert captured.out == "0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n"

    def test_is_multiple_of_pipeline(self, capsys):
        generate_numbers_pipe = GenerateNumbersPipe(10)
        is_multiple_of_pipe = IsMultipleOfPipe(factor=2)
        print_pipe = PrintPipe()

        # Create pipeline
        pipeline = Pipeline(generate_numbers_pipe)
        pipeline.filter(is_multiple_of_pipe)
        pipeline.map(print_pipe)

        pipeline.run()

        captured = capsys.readouterr()

        assert captured.out == "0\n2\n4\n6\n8\n"

    def test_moving_average_pipeline(self, capsys):
        generate_numbers_pipe = GenerateNumbersPipe(10)
        moving_average_pipe = MovingAveragePipe(period=5)
        print_pipe = PrintPipe(key="avg")

        # Create pipeline
        pipeline = Pipeline(generate_numbers_pipe)
        pipeline.map(moving_average_pipe)
        pipeline.filter(lambda data: "avg" in data)
        pipeline.map(print_pipe)

        pipeline.run()

        captured = capsys.readouterr()

        assert captured.out == "2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n"

    def test_batch_pipeline(self, capsys):
        generate_numbers_pipe = GenerateNumbersPipe(10)
        batch_pipe = BatchPipe(batch_size=2)
        print_pipe = PrintPipe(key="batch_no")

        # Create pipeline
        pipeline = Pipeline(generate_numbers_pipe)
        pipeline.iter(batch_pipe)
        pipeline.map(print_pipe)

        pipeline.run()

        captured = capsys.readouterr()

        assert captured.out == "1\n1\n2\n2\n3\n3\n4\n4\n5\n5\n"