from time import time
from .coloured import Print


def timer(func):
    """decorator to time functions"""
    def func_timer(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        Print(f'b([INFO]) w(Measure: {func.__name__} was executed in {round(end - start), 2}s)')
        return result
    return func_timer


def counter(num=1, length=3):
    """Counter etc. 0001, 0002
    Attributes:
    num (int) integer etc. 1 ==> 0001
        length (int) length of counter etc. 3 ==> 001
    Return:
        (str) etc. 0001
    """
    number = '0' * length + str(num)
    number = number[len(number)-length:]
    return number