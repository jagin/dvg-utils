import logging
import time
import functools


def timeit(func):
    """Python decorator to measure the execution time of methods.

    Example usage::

        from dvgutils import timeit

        @timeit
        def my_method():
            ...

    """
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.perf_counter()
        result = func(*args, **kw)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000.0

        logging.getLogger(f"{func.__module__}.{func.__name__}").debug(f"{func.__module__}.{func.__name__} execution time: {elapsed_time:0.4f} ms")

        return result

    return wrapper
