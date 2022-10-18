import time


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        finish = time.time()
        execution_time = finish - start
        print('Execution time: ', execution_time)
        return execution_time
    return wrapper

