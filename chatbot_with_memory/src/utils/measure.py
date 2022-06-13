from time import time
from .coloured import Print


def timer(func):
    """decorator to time functions"""
    def func_timer(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        Print(f'm([INFO]) w(Measure: {func.__name__} was executed in {round(end - start, 2)}s)')
        return result
    return func_timer
