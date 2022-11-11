import time


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        finish = time.time()
        execution_time = finish - start
        #print('Execution time: ', execution_time)
        return {'execution_time': execution_time, 'func_res': res}
    return wrapper
