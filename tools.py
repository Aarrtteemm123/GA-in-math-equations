import json
import time


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        finish = time.time()
        execution_time = finish - start
        # print('Execution time: ', execution_time)
        return {'execution_time': execution_time, 'func_res': res}

    return wrapper


def read_json_from_file(path):
    with open(path, 'r') as file:
        return json.loads(file.read())
