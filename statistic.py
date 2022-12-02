import os
import numpy as np
import json
import sympy
from multiprocessing import Process
from sympy import Symbol, solve, Pow, exp, cos, tan
from ga_interface import GAInterface
from tools import benchmark, read_json_from_file
from numpy import arange
from scipy.optimize import curve_fit
from matplotlib import pyplot
from config import Config


class AnalyserGA:
    def __init__(self, attempts, path_to_result):
        self.attempts = attempts
        self.result = {}
        self.path_to_result = path_to_result

    def linear_equation(self, a, b, save=False):
        equation = f'{a}*x+{b}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=8,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['linear_equation'] = result
        if save:
            self.save(self.path_to_result + 'linear_equation.json', result)

    def sqrt_x(self, a, b, c, save=False):
        equation = f'{a}*math.sqrt(x*{b})+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.05,
                        gene_space={"low": 1, "high": 30},
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result = result
        if save:
            self.save(self.path_to_result + 'sqrt_x.json')

    def polynomial_2(self, a, b, c, save=False):
        equation = f'{a}*(x**2)+{b}*x+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.2,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_2'] = result
        if save:
            self.save(self.path_to_result + 'polynomial_2.json', result)

    def polynomial_3(self, a, b, c, d, save=False):
        equation = f'{a}*(x**3)+{b}*(x**2)+{c}*x+{d}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=3,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.2,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_3'] = result
        if save:
            self.save(self.path_to_result + 'polynomial_3.json', result)

    def polynomial_4(self, a, b, c, d, e, save=False):
        equation = f'{a}*(x**4)+{b}*(x**3)+{c}*(x**2)+{d}*x+{e}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=3,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.2,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_4'] = result
        if save:
            self.save(self.path_to_result + 'polynomial_4.json', result)

    def polynomial_5(self, a, b, c, d, e, f, save=False):
        equation = f'{a}*(x**5)+{b}*(x**4)+{c}*(x**3)+{d}*(x**2)+{e}*x+{f}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.2,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_5'] = result
        if save:
            self.save(self.path_to_result + 'polynomial_5.json', result)

    def exponential_equation(self, a, b, c, save=False):
        equation = f'{a}*(math.e**(x*{b}))+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.18,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['exponential_equation'] = result
        if save:
            self.save(self.path_to_result + 'exponential_equation.json', result)

    def sin_x(self, a, b, c, save=False):
        equation = f'{a}*math.sin({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.18,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['sin_x'] = result
        if save:
            self.save(self.path_to_result + 'sin_x.json', result)

    def cos_x(self, a, b, c, save=False):
        equation = f'{a}*math.cos({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=10,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.22,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['cos_x'] = result
        if save:
            self.save(self.path_to_result + 'cos_x.json', result)

    def tg_x(self, a, b, c, save=False):
        equation = f'{a}*math.tan({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=3,
                        sol_per_pop=50,
                        num_genes=12,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.35,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['tg_x'] = result
        if save:
            self.save(self.path_to_result + 'tg_x.json', result)

    def ctg_x(self, a, b, c, save=False):
        equation = f'{a}/math.tan({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=3,
                        sol_per_pop=50,
                        num_genes=12,
                        accuracy=0.05,
                        crossover_type='two_points',
                        mutation_probability=0.35,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['ctg_x'] = result
        if save:
            self.save(self.path_to_result + 'ctg_x.json', result)

    def save(self, path_to_result, data=None):
        with open(path_to_result, 'w') as file:
            file.write(json.dumps(data if data else self.result, indent=4))

    def _run(self, ga):
        execution_times, generations, ga_result, errors = [], [], {}, []
        fails = 0
        for _ in range(self.attempts):
            execution_time = ga.run_solver()['execution_time']
            ga_result = ga.get_result()
            if ga_result.get('error') > ga_result.get('accuracy'):
                fails += 1
                continue
            execution_times.append(execution_time)
            generations.append(ga_result.pop('generations_completed'))
            errors.append(ga_result.get('error'))

        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        avg_generations_completed = sum(generations) / len(generations) if generations else 0
        avg_error = sum(errors) / len(errors) if errors else 0
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'avg_generations_completed': avg_generations_completed,
            'avg_error': avg_error,
            'attempts': self.attempts,
            'fails': fails,
            'percent_fails': 100 * fails / self.attempts,
            **ga_result
        }


class AnalyzerComputerAlgebra:
    def __init__(self):
        self.x = Symbol('x')
        self.result = {}

    @benchmark
    def _run(self, func, *args):
        res_lst = func(*args)
        return [float(res) for res in res_lst if complex(res).imag == 0]

    def linear_equation(self, a, b):
        res = self._run(solve, a * self.x + b, self.x)
        self.result['linear_equation'] = res

    def sqrt_x(self, a, b, c):
        res = self._run(solve, a * Pow(self.x * b, 0.5) + c, self.x)
        self.result['sqrt_x'] = res

    def polynomial_2(self, a, b, c):
        res = self._run(solve, a * Pow(self.x, 2) + b * self.x + c, self.x)
        self.result['polynomial_2'] = res

    def polynomial_3(self, a, b, c, d):
        res = self._run(solve, a * Pow(self.x, 3) + b * Pow(self.x, 2) + c * self.x + d, self.x)
        self.result['polynomial_3'] = res

    def polynomial_4(self, a, b, c, d, e):
        res = self._run(solve, a * Pow(self.x, 4) + b * Pow(self.x, 3) + c * Pow(self.x, 2) + d * self.x + e, self.x)
        self.result['polynomial_4'] = res

    def polynomial_5(self, a, b, c, d, e, f):
        res = self._run(solve, a * Pow(self.x, 5) + b * Pow(self.x, 4) + c * Pow(self.x, 3) + d * Pow(self.x,
                                                                                                      2) + e * self.x + f,
                        self.x)
        self.result['polynomial_5'] = res

    def exponential_equation(self, a, b, c):
        res = self._run(solve, a * exp(b * self.x) + c, self.x)
        self.result['exponential_equation'] = res

    def sin_x(self, a, b, c):
        res = self._run(solve, a * cos((sympy.pi / 2) - b * self.x) + c, self.x)
        self.result['sin_x'] = res

    def cos_x(self, a, b, c):
        res = self._run(solve, a * cos(b * self.x) + c, self.x)
        self.result['cos_x'] = res

    def tg_x(self, a, b, c):
        res = self._run(solve, a * tan(b * self.x) + c, self.x)
        self.result['tg_x'] = res

    def ctg_x(self, a, b, c):
        res = self._run(solve, a / tan(b * self.x) + c, self.x)
        self.result['ctg_x'] = res

    def save(self, path_to_result):
        with open(path_to_result + 'AnalyzerComputerAlgebra.json', 'w') as file:
            file.write(json.dumps(self.result, indent=4))


class LinerEquationAnalyzerGA:
    def __init__(self, attempts, a, b, path_to_result):
        self.attempts = attempts
        self.a = a
        self.b = b
        self.equation = f'{self.a}*x+{self.b}'
        self.result = {}
        self.path_to_result = path_to_result

    def analyze_generations(self, save=False):
        ga = GAInterface(self.equation)
        ga_const_config = {
            'num_parents_mating': 2,
            'sol_per_pop': 20,
            'num_genes': 6,
            'accuracy': 0.005,
            'crossover_type': 'single_point',
            'mutation_probability': 0.1,
            'parallel_processing': 1
        }
        self.result['analyze_generation'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                                 generations=[])

        for num_generation in range(10, 1010, 10):
            print('Generation:', num_generation)
            ga.build_solver(num_generations=num_generation, **ga_const_config)
            result = self._run(ga)
            needed_data = {
                'num_generations': num_generation,
                'avg_generation': result.get('avg_generations_completed'),
                'avg_error': result.get('avg_error'),
                'error': result.get('error'),
                'fails': result.get('fails'),
                'percent_fails': result.get('percent_fails'),
                'avg_time': result.get('avg_time')
            }
            self.result['analyze_generation']['generations'].append(needed_data)

        if save:
            self.save(self.path_to_result + 'analyze_generations.json', self.result['analyze_generation'])

    def analyze_population_size(self, save=False):
        ga = GAInterface(self.equation)
        ga_const_config = {
            'num_generations': 1000,
            'num_parents_mating': 2,
            'num_genes': 4,
            'accuracy': 0.01,
            'crossover_type': 'uniform',
            'mutation_probability': 0.1,
            'parallel_processing': 1
        }
        self.result['analyze_population_size'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                                      population_size=[])

        for population_size in range(4, 200, 1):
            print('Pop.size:', population_size)
            ga.build_solver(sol_per_pop=population_size, **ga_const_config)
            result = self._run(ga)
            needed_data = {
                'population_size': population_size,
                'avg_generation': result.get('avg_generations_completed'),
                'avg_error': result.get('avg_error'),
                'error': result.get('error'),
                'fails': result.get('fails'),
                'percent_fails': result.get('percent_fails'),
                'avg_time': result.get('avg_time')
            }
            self.result['analyze_population_size']['population_size'].append(needed_data)

        if save:
            self.save(self.path_to_result + 'analyze_population_size.json', self.result['analyze_population_size'])

    def analyze_num_genes(self, save=False):
        ga = GAInterface(self.equation)
        ga_const_config = {
            'num_generations': 1000,
            'num_parents_mating': 2,
            'sol_per_pop': 20,
            'accuracy': 0.01,
            'crossover_type': 'single_point',
            'mutation_probability': 0.15,
            'parallel_processing': 1
        }
        self.result['analyze_num_genes'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                                num_genes=[])

        for num_genes in range(2, 100, 1):
            print('Num.genes:', num_genes)
            ga.build_solver(num_genes=num_genes, **ga_const_config)
            result = self._run(ga)
            needed_data = {
                'num_genes': num_genes,
                'avg_generation': result.get('avg_generations_completed'),
                'avg_error': result.get('avg_error'),
                'error': result.get('error'),
                'fails': result.get('fails'),
                'percent_fails': result.get('percent_fails'),
                'avg_time': result.get('avg_time')
            }
            self.result['analyze_num_genes']['num_genes'].append(needed_data)

        if save:
            self.save(self.path_to_result + 'analyze_num_genes.json', self.result['analyze_num_genes'])

    def analyze_crossover_types(self, save=False):
        ga = GAInterface(self.equation)
        crossover_types = ['single_point', 'two_points', 'uniform', 'scattered']
        ga_const_config = {
            'num_generations': 5000,
            'num_parents_mating': 2,
            'num_genes': 5,
            'sol_per_pop': 10,
            'accuracy': 0.01,
            'mutation_probability': 0.15,
            'parallel_processing': 1
        }
        self.result['analyze_crossover_types'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                                      crossover_types=[])

        for crossover_type in crossover_types:
            print('Crossover type:', crossover_type)
            ga.build_solver(crossover_type=crossover_type, **ga_const_config)
            result = self._run(ga)
            needed_data = {
                'crossover_type': crossover_type,
                'num_genes': result.get('num_genes'),
                'avg_generation': result.get('avg_generations_completed'),
                'avg_error': result.get('avg_error'),
                'error': result.get('error'),
                'fails': result.get('fails'),
                'percent_fails': result.get('percent_fails'),
                'avg_time': result.get('avg_time')
            }
            self.result['analyze_crossover_types']['crossover_types'].append(needed_data)

        if save:
            self.save(self.path_to_result + 'analyze_crossover_types.json', self.result['analyze_crossover_types'])

    def analyze_mutation_probability(self, save=False):
        ga = GAInterface(self.equation)
        ga_const_config = {
            'num_generations': 1000,
            'num_parents_mating': 2,
            'num_genes': 4,
            'sol_per_pop': 10,
            'accuracy': 0.01,
            'crossover_type': 'two_points',
            'parallel_processing': 1
        }
        self.result['analyze_mutation_probability'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                                           mutation_probabilities=[])

        for value in range(100):
            mutation_probability = value / 100
            print('Mutation probability:', mutation_probability)
            ga.build_solver(mutation_probability=mutation_probability, **ga_const_config)
            result = self._run(ga)
            needed_data = {
                'mutation_probability': mutation_probability,
                'avg_generation': result.get('avg_generations_completed'),
                'avg_error': result.get('avg_error'),
                'error': result.get('error'),
                'fails': result.get('fails'),
                'percent_fails': result.get('percent_fails'),
                'avg_time': result.get('avg_time')
            }
            self.result['analyze_mutation_probability']['mutation_probabilities'].append(needed_data)

        if save:
            self.save(self.path_to_result + 'analyze_mutation_probability.json',
                      self.result['analyze_mutation_probability'])

    def analyze_accuracy(self, save=False):
        ga = GAInterface(self.equation)
        ga_const_config = {
            'num_generations': 1000,
            'num_parents_mating': 2,
            'num_genes': 6,
            'sol_per_pop': 20,
            'crossover_type': 'two_points',
            'mutation_probability': 0.2,
            'parallel_processing': 1,
        }
        self.result['analyze_accuracy'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                               accuracy=[])

        for accuracy in np.arange(0.001, 0.5, 0.001):
            print('Accuracy:', accuracy)
            ga.build_solver(accuracy=accuracy, **ga_const_config)
            result = self._run(ga)
            needed_data = {
                'accuracy': accuracy,
                'avg_generation': result.get('avg_generations_completed'),
                'avg_error': result.get('avg_error'),
                'error': result.get('error'),
                'fails': result.get('fails'),
                'percent_fails': result.get('percent_fails'),
                'avg_time': result.get('avg_time')
            }
            self.result['analyze_accuracy']['accuracy'].append(needed_data)

        if save:
            self.save(self.path_to_result + 'analyze_accuracy.json', self.result['analyze_accuracy'])

    def _run(self, ga):
        execution_times, generations, ga_result, errors = [], [], {}, []
        fails = 0
        for _ in range(self.attempts):
            execution_time = ga.run_solver()['execution_time']
            ga_result = ga.get_result()
            if ga_result.get('error') > ga_result.get('accuracy'):
                fails += 1
                continue
            execution_times.append(execution_time)
            generations.append(ga_result.pop('generations_completed'))
            errors.append(ga_result.get('error'))

        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        avg_generations_completed = sum(generations) / len(generations) if generations else 0
        avg_error = sum(errors) / len(errors) if errors else 0
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'avg_generations_completed': avg_generations_completed,
            'avg_error': avg_error,
            'attempts': self.attempts,
            'fails': fails,
            'percent_fails': 100 * fails / self.attempts,
            **ga_result
        }

    def save(self, path_to_result, data=None):
        with open(path_to_result, 'w') as file:
            file.write(json.dumps(data if data else self.result, indent=4))


def build_graphs_line(data_x, data_y, func_type='pol'):
    """func type: pol, exp, log, lin"""

    def fit_func(x, a, b, c, d, e):
        if func_type == 'pol':
            return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
        if func_type == 'exp':
            return (a - c) * np.exp(-x / b) + c
        if func_type == 'log':
            return a * np.log(b * x) + c
        if func_type == 'lin':
            return a * x + b
        raise Exception('func type not found, choose one of next ["pol", "exp", "log", "lin"]')

    popt, _ = curve_fit(fit_func, data_x, data_y, maxfev=8000)
    x_line = arange(min(data_x), max(data_x), 0.01)
    y_line = fit_func(x_line, *popt)
    return x_line, y_line


def create_plot(path_to_save, data_x, data_y, title, label_x, label_y, fit_func='pol', scale_x=False, scale_y=False,
                include_fit_func=True, scatter=False):
    x_line, y_line = build_graphs_line(data_x, data_y, fit_func)
    fig, ax = pyplot.subplots()
    if scale_y:
        ax.set_yscale('log')
    if scale_x:
        ax.set_xscale('log')

    if scatter:
        ax.scatter(data_x, data_y)
    if not scatter:
        ax.plot(data_x, data_y)

    if include_fit_func:
        ax.plot(x_line, y_line, '--', color='red')
    pyplot.title(title)
    pyplot.xlabel(label_x)
    pyplot.ylabel(label_y)
    pyplot.draw()
    pyplot.savefig(path_to_save)


def create_bar(path_to_save, data_x, data_y, title, label_x, label_y):
    fig, ax = pyplot.subplots()
    ax.bar(data_x, data_y)
    pyplot.title(title)
    pyplot.xlabel(label_x)
    pyplot.ylabel(label_y)
    pyplot.draw()
    pyplot.savefig(path_to_save)


def create_multibar(path_to_save, min_time, avg_time, max_time, comp_alg_time, title):
    x = np.arange(1)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = pyplot.subplots()
    rects1 = ax.bar(x - 0.5 * width, round(min_time, 2), width, color='g', alpha=0.6, label='min ga')
    rects2 = ax.bar(x + 0.5 * width, round(avg_time, 2), width, color='b', alpha=0.6, label='avg ga')
    rects3 = ax.bar(x + 1.5 * width, round(max_time, 2), width, color='r', alpha=0.6, label='max ga')
    rects4 = ax.bar(x + 4 * width, round(comp_alg_time, 2), width, alpha=0.6, color='m', label='comp.alg')

    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    ax.set_xticks([], [])
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    fig.tight_layout()
    pyplot.draw()
    pyplot.savefig(path_to_save)


def get_formatted_data(path, list_key, x_data_key):
    data = read_json_from_file(path)
    data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time = [], [], [], [], []
    for item in data[list_key]:
        data_x.append(item[x_data_key])
        data_avg_gen.append(item['avg_generation'])
        data_avg_error.append(item['avg_error'])
        data_per_fails.append(item['percent_fails'])
        data_avg_time.append(item['avg_time'] * 1000)
    return data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time


def get_multibar_formatted_data(file, data_comp_alg):
    data_ga = read_json_from_file(Config.PATH_TO_STATISTIC_DATA + file)
    avg_time, min_time, max_time = data_ga['avg_time'] * 1000, data_ga['min_time'] * 1000, data_ga['max_time'] * 1000
    accuracy = data_ga['accuracy']
    comp_alg_time = data_comp_alg[file.removesuffix('.json')]['execution_time'] * 1000
    return min_time, avg_time, max_time, comp_alg_time, accuracy


def create_images():
    files = os.listdir(Config.PATH_TO_STATISTIC_DATA)
    data_comp_alg = read_json_from_file(Config.PATH_TO_STATISTIC_DATA + 'AnalyzerComputerAlgebra.json')
    for file in files:
        if file == 'analyze_accuracy.json':
            data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time = get_formatted_data(
                Config.PATH_TO_STATISTIC_DATA + file, 'accuracy', 'accuracy')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'accuracy-avg_generation.png', data_x, data_avg_gen,
                        'Average generations vs accuracy', 'accuracy', 'avg.generation', 'exp')
            create_plot(Config.PATH_TO_STATISTIC_IMG + 'accuracy-avg_generation2.png', data_x, data_avg_gen,
                        'Average generations vs accuracy',
                        'accuracy', 'avg.generation', 'exp', scale_y=True, include_fit_func=False)

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'accuracy-error.png', data_x, data_avg_error,
                        'Average error vs accuracy', 'accuracy', 'avg.error', 'exp')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'accuracy-fails.png', data_x, data_per_fails,
                        'Percent fails vs accuracy', 'accuracy', 'fails (%)', 'lin')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'accuracy-time.png', data_x, data_avg_time,
                        'Average time vs accuracy', 'accuracy', 'avg.time (ms)', 'exp')
            create_plot(Config.PATH_TO_STATISTIC_IMG + 'accuracy-time2.png', data_x, data_avg_time,
                        'Average time vs accuracy', 'accuracy', 'avg.time (ms)', scale_y=True, include_fit_func=False)

        elif file == 'analyze_crossover_types.json':
            data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time = get_formatted_data(
                Config.PATH_TO_STATISTIC_DATA + file, 'crossover_types', 'crossover_type')

            create_bar(Config.PATH_TO_STATISTIC_IMG + 'crossover_type-avg_generation.png', data_x, data_avg_gen,
                       'Average generations vs crossover type', 'crossover type', 'avg.generation')

            create_bar(Config.PATH_TO_STATISTIC_IMG + 'crossover_type-avg_error.png', data_x, data_avg_error,
                       'Average error vs crossover type', 'crossover type', 'avg.error')

            create_bar(Config.PATH_TO_STATISTIC_IMG + 'crossover_type-per_fails.png', data_x, data_per_fails,
                       'Fails vs crossover type', 'crossover type', 'fails')

            create_bar(Config.PATH_TO_STATISTIC_IMG + 'crossover_type-avg_time.png', data_x, data_avg_time,
                       'Average time vs crossover type', 'crossover type', 'avg.time (ms)')

        elif file == 'analyze_generations.json':
            data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time = get_formatted_data(
                Config.PATH_TO_STATISTIC_DATA + file, 'generations', 'num_generations')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'generations-error.png', data_x, data_avg_error,
                        'Average error vs generations', 'generations', 'avg.error', 'lin')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'generations-fails.png', data_x, data_per_fails,
                        'Percent fails vs generations', 'generations', 'fails (%)', 'exp')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'generations-time.png', data_x, data_avg_time,
                        'Average time vs generations', 'generations', 'avg.time (ms)', 'exp')


        elif file == 'analyze_mutation_probability.json':
            data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time = get_formatted_data(
                Config.PATH_TO_STATISTIC_DATA + file, 'mutation_probabilities', 'mutation_probability')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'mutation_probabilities-avg_generation.png', data_x,
                        data_avg_gen,
                        'Average generations vs mutation probabilities', 'mutation probability', 'avg.generation',
                        'exp')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'mutation_probabilities-per_fails.png', data_x, data_per_fails,
                        'Percent fails vs mutation probabilities', 'mutation probability', 'fails (%)', 'exp')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'mutation_probabilities-avg_time.png', data_x, data_avg_time,
                        'Average time vs mutation probabilities', 'mutation probability', 'avg.time (ms)', 'exp',
                        scatter=True)

        elif file == 'analyze_num_genes.json':
            data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time = get_formatted_data(
                Config.PATH_TO_STATISTIC_DATA + file, 'num_genes', 'num_genes')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'num_genes-avg_generation.png', data_x,
                        data_avg_gen,
                        'Average generations vs number genes', 'number genes', 'avg.generation',
                        'pol')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'num_genes-avg_error.png', data_x,
                        data_avg_error,
                        'Average error vs number genes', 'number genes', 'avg.error',
                        'pol')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'num_genes-per_fails.png', data_x, data_per_fails,
                        'Percent fails vs number genes', 'number genes', 'fails (%)', 'pol')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'num_genes-avg_time.png', data_x, data_avg_time,
                        'Average time vs number genes', 'number genes', 'avg.time (ms)', 'pol',
                        scatter=False)

        elif file == 'analyze_population_size.json':
            data_x, data_avg_gen, data_avg_error, data_per_fails, data_avg_time = get_formatted_data(
                Config.PATH_TO_STATISTIC_DATA + file, 'population_size', 'population_size')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'population_size-avg_generation.png', data_x,
                        data_avg_gen,
                        'Average generations vs population size', 'population size', 'avg.generation',
                        'exp')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'population_size-avg_error.png', data_x,
                        data_avg_error,
                        'Average error vs population size', 'population size', 'avg.error',
                        'lin')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'population_size-per_fails.png', data_x, data_per_fails,
                        'Percent fails vs population size', 'population size', 'fails (%)', 'log')

            create_plot(Config.PATH_TO_STATISTIC_IMG + 'population_size-avg_time.png', data_x, data_avg_time,
                        'Average time vs population size', 'population size', 'avg.time (ms)', 'log',
                        scatter=False)

        elif file == 'cos_x.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve cos equation (acc={accuracy})')
        elif file == 'ctg_x.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve ctg equation (acc={accuracy})')
        elif file == 'exponential_equation.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve exponential equation (acc={accuracy})')
        elif file == 'linear_equation.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve linear equation (acc={accuracy})')

        elif file == 'polynomial_2.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve n=2 polynomial equation (acc={accuracy})')
        elif file == 'polynomial_3.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve n=3 polynomial equation (acc={accuracy})')
        elif file == 'polynomial_4.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve n=4 polynomial equation (acc={accuracy})')
        elif file == 'polynomial_5.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve n=5 polynomial equation (acc={accuracy})')
        elif file == 'sin_x.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve sin equation (acc={accuracy})')
        elif file == 'sqrt_x.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve square root equation (acc={accuracy})')
        elif file == 'tg_x.json':
            min_time, avg_time, max_time, comp_alg_time, accuracy = get_multibar_formatted_data(file, data_comp_alg)

            create_multibar(Config.PATH_TO_STATISTIC_IMG + file.replace('.json', '.png'), min_time, avg_time, max_time,
                            comp_alg_time, f'Speed solve tg equation (acc={accuracy})')


def run_statistic():
    analyser = AnalyserGA(Config.ATTEMPTS, Config.PATH_TO_STATISTIC_DATA)
    # analyser.linear_equation(5, -3, save=True)
    # analyser.sqrt_x(4, 2, -50, save=True)
    # analyser.polynomial_2(2, 5, -15, save=True)
    # analyser.polynomial_3(10, -1, -10, 4, save=True)
    # analyser.polynomial_4(-4, -7, 5, 4, -2, save=True)
    # analyser.polynomial_5(-3, -4, -7, 5, 5, 1, save=True)
    # analyser.exponential_equation(5, 1, -10, save=True)
    # analyser.sin_x(4, -2, 2, save=True)
    # analyser.cos_x(7, 2, 3, save=True)
    # analyser.tg_x(-5, 3, -2, save=True)
    # analyser.ctg_x(4, 8, -10, save=True)
    # Process(target=analyser.linear_equation, args=(5, -3, True)).start()
    # Process(target=analyser.sqrt_x, args=(4, 2, -50, True)).start()
    # Process(target=analyser.polynomial_2, args=(2, 5, -15, True)).start()
    # Process(target=analyser.polynomial_3, args=(10, -1, -10, 4, True)).start()
    # Process(target=analyser.polynomial_4, args=(-4, -7, 5, 4, -2, True)).start()
    # Process(target=analyser.polynomial_5, args=(-3, -4, -7, 5, 5, 1, True)).start()
    # Process(target=analyser.exponential_equation, args=(5, 1, -10, True)).start()
    # Process(target=analyser.sin_x, args=(4, -2, 2, True)).start()
    # Process(target=analyser.cos_x, args=(7, 2, 3, True)).start()
    # Process(target=analyser.tg_x, args=(-5, 3, -2, True)).start()
    # Process(target=analyser.ctg_x, args=(4, 8, -10, True)).start()

    # anal_comp_alg = AnalyzerComputerAlgebra()
    # anal_comp_alg.linear_equation(5, -3)
    # anal_comp_alg.sqrt_x(4, 2, -50)
    # anal_comp_alg.polynomial_2(2, 5, -15)
    # anal_comp_alg.polynomial_3(10, -1, -10, 4)
    # anal_comp_alg.polynomial_4(-4, -7, 5, 4, -2)
    # anal_comp_alg.polynomial_5(-3, -4, -7, 5, 5, 1)
    # anal_comp_alg.exponential_equation(5, 1, -10)
    # anal_comp_alg.sin_x(4, -2, 2)
    # anal_comp_alg.cos_x(7, 2, 3)
    # anal_comp_alg.tg_x(-5, 3, -2)
    # anal_comp_alg.ctg_x(4, 8, -10)
    # anal_comp_alg.save(Config.PATH_TO_STATISTIC_DATA)

    liner_analyzer = LinerEquationAnalyzerGA(Config.ATTEMPTS, 5, -3, Config.PATH_TO_STATISTIC_DATA)
    # Process(target=liner_analyzer.analyze_generations, args=(True,)).start()
    # Process(target=liner_analyzer.analyze_population_size, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_num_genes, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_crossover_types, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_mutation_probability, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_accuracy, args=(True, )).start()


if __name__ == '__main__':
    #run_statistic()
    create_images()
