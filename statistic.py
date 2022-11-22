import json
import math

import numpy
from sympy import Symbol, solve, Pow, exp, sin, cos, tan

from config import Config
from ga_interface import GAInterface
from tools import benchmark

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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
                        sol_per_pop=100,
                        num_genes=8,
                        accuracy=0.05,
                        crossover_type='single_point',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['linear_equation'] = result
        if save:
            self.save(self.path_to_result+'linear_equation.json', result)

    def sqrt_x(self, a, b, c, save=False):
        equation = f'{a}*math.sqrt(x*{b})+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=50,
                        num_genes=8,
                        accuracy=0.01,
                        gene_space={"low": 1, "high": 30},
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['sqrt_x'] = result
        if save:
            self.save(self.path_to_result+'sqrt_x.json')

    def polynomial_2(self, a, b, c, save=False):
        equation = f'{a}*(x**2)+{b}*x+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=10,
                        num_genes=6,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.2,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_2'] = result
        if save:
            self.save(self.path_to_result+'polynomial_2.json', result)

    def polynomial_3(self, a, b, c, d, save=False):
        equation = f'{a}*(x**3)+{b}*(x**2)+{c}*x+{d}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=3,
                        sol_per_pop=10,
                        num_genes=8,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.2,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_3'] = result
        if save:
            self.save(self.path_to_result+'polynomial_3.json', result)

    def polynomial_4(self, a, b, c, d, e, save=False):
        equation = f'{a}*(x**4)+{b}*(x**3)+{c}*(x**2)+{d}*x+{e}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=3,
                        sol_per_pop=10,
                        num_genes=10,
                        accuracy=0.1,
                        crossover_type='single_point',
                        mutation_probability=0.2,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_4'] = result
        if save:
            self.save(self.path_to_result+'polynomial_4.json', result)

    def polynomial_5(self, a, b, c, d, e, f, save=False):
        equation = f'{a}*(x**5)+{b}*(x**4)+{c}*(x**3)+{d}*(x**2)+{e}*x+{f}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=10,
                        num_genes=10,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['polynomial_5'] = result
        if save:
            self.save(self.path_to_result+'polynomial_5.json', result)

    def exponential_equation(self, a, b, c, save=False):
        equation = f'{a}*(math.e**(x*{b}))+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=20,
                        num_genes=10,
                        accuracy=0.01,
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['exponential_equation'] = result
        if save:
            self.save(self.path_to_result+'exponential_equation.json', result)

    def sin_x(self, a, b, c, save=False):
        equation = f'{a}*math.sin({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=10,
                        num_genes=10,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['sin_x'] = result
        if save:
            self.save(self.path_to_result+'sin_x.json', result)

    def cos_x(self, a, b, c, save=False):
        equation = f'{a}*math.cos({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=10,
                        num_genes=10,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['cos_x'] = result
        if save:
            self.save(self.path_to_result + 'cos_x.json', result)

    def tg_x(self, a, b, c, save=False):
        equation = f'{a}*math.tan({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=10,
                        num_genes=10,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.15,
                        parallel_processing=1)
        result = self._run(ga)
        self.result['tg_x'] = result
        if save:
            self.save(self.path_to_result + 'tg_x.json', result)

    def ctg_x(self, a, b, c, save=False):
        equation = f'{a}/math.tan({b}*x)+{c}'
        ga = GAInterface(equation)
        ga.build_solver(num_generations=1000,
                        num_parents_mating=2,
                        sol_per_pop=10,
                        num_genes=10,
                        accuracy=0.1,
                        crossover_type='two_points',
                        mutation_probability=0.15,
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
        res = self._run(solve, a * sin(b * self.x) + c, self.x)
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
            print(num_generation)
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
            'num_genes': 6,
            'accuracy': 0.01,
            'crossover_type': 'single_point',
            'mutation_probability': 0.15,
            'parallel_processing': 1
        }
        self.result['analyze_population_size'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                                      population_size=[])

        for population_size in range(4, 200, 1):
            print(population_size)
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
            print(num_genes)
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
            print(crossover_type)
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
            print(mutation_probability)
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
            self.save(self.path_to_result + 'analyze_mutation_probability.json', self.result['analyze_mutation_probability'])

    def analyze_accuracy(self, save=False):
        ga = GAInterface(self.equation)
        ga_const_config = {
            'num_generations': 1000,
            'num_parents_mating': 2,
            'num_genes': 6,
            'sol_per_pop': 50,
            'crossover_type': 'two_points',
            'mutation_probability': 0.2,
            'parallel_processing': 1,
        }
        self.result['analyze_accuracy'] = dict(base_config={'attempts': self.attempts, **ga_const_config},
                                               accuracy=[])

        for accuracy in np.arange(0.001, 1.0, 0.001):
            print(accuracy)
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


def build_graphs():
    pass


def run_statistic():
    # analyser = AnalyserGA(Config.ATTEMPTS, Config.PATH_TO_STATISTIC)
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
    # anal_comp_alg.save(Config.PATH_TO_STATISTIC)

    liner_analyzer = LinerEquationAnalyzerGA(Config.ATTEMPTS, 5, -3, Config.PATH_TO_STATISTIC)
    from multiprocessing import Process
    # Process(target=liner_analyzer.analyze_generations, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_population_size, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_num_genes, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_crossover_types, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_mutation_probability, args=(True, )).start()
    # Process(target=liner_analyzer.analyze_accuracy, args=(True, )).start()
    # liner_analyzer.analyze_generations(save=True)
    # liner_analyzer.analyze_population_size(save=True)
    # liner_analyzer.analyze_num_genes(save=True)
    # liner_analyzer.analyze_crossover_types(save=True)
    # liner_analyzer.analyze_mutation_probability(save=True)
    # liner_analyzer.analyze_accuracy(save=True)


if __name__ == '__main__':
    run_statistic()
    build_graphs()
