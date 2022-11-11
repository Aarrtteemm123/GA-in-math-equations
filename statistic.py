import json

from sympy import Symbol, solve, Pow, exp, sin, cos, tan

from config import Config
from ga_interface import GAInterface
from tools import benchmark


class AnalyserGA:
    def __init__(self, num_loop):
        self.num_loop = num_loop
        self.result = {}

    def linear_equation(self, a, b):
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

    def sqrt_x(self, a, b, c):
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

    def polynomial_2(self, a, b, c):
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

    def polynomial_3(self, a, b, c, d):
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

    def polynomial_4(self, a, b, c, d, e):
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

    def polynomial_5(self, a, b, c, d, e, f):
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

    def exponential_equation(self, a, b, c):
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

    def sin_x(self, a, b, c):
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

    def cos_x(self, a, b, c):
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

    def tg_x(self, a, b, c):
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

    def ctg_x(self, a, b, c):
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

    def save(self, path_to_result):
        with open(path_to_result + 'AnalyserGA.json', 'w') as file:
            file.write(json.dumps(self.result, indent=4))

    def _run(self, ga):
        execution_times, generations, ga_result = [], [], {}
        fails = 0
        for _ in range(self.num_loop):
            execution_time = ga.run_solver()['execution_time']
            ga_result = ga.get_result()
            if ga_result.get('error') > ga_result.get('accuracy'):
                fails += 1
                continue
            execution_times.append(execution_time)
            generations.append(ga_result.pop('generations_completed'))

        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        avg_generations_completed = sum(generations) / len(generations) if generations else 0
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'avg_generations_completed': avg_generations_completed,
            'attempts': self.num_loop,
            'fails': fails,
            'percent_fails': 100 * fails / self.num_loop,
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
        res = self._run(solve, a * Pow(self.x, 5) + b * Pow(self.x, 4) + c * Pow(self.x, 3) + d * Pow(self.x, 2) + e * self.x + f, self.x)
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


def run_statistic():
    analyser = AnalyserGA(Config.NUM_LOOPS)
    analyser.linear_equation(5, -3)
    analyser.sqrt_x(4, 2, -50)
    analyser.polynomial_2(2, 5, -15)
    analyser.polynomial_3(10, -1, -10, 4)
    analyser.polynomial_4(-4, -7, 5, 4, -2)
    analyser.polynomial_5(-3, -4, -7, 5, 5, 1)
    analyser.exponential_equation(5, 1, -10)
    analyser.sin_x(4, -2, 2)
    analyser.cos_x(7, 2, 3)
    analyser.tg_x(-5, 3, -2)
    analyser.ctg_x(4, 8, -10)
    analyser.save(Config.PATH_TO_STATISTIC)

    anal_comp_alg = AnalyzerComputerAlgebra()
    anal_comp_alg.linear_equation(5, -3)
    anal_comp_alg.sqrt_x(4, 2, -50)
    anal_comp_alg.polynomial_2(2, 5, -15)
    anal_comp_alg.polynomial_3(10, -1, -10, 4)
    anal_comp_alg.polynomial_4(-4, -7, 5, 4, -2)
    anal_comp_alg.polynomial_5(-3, -4, -7, 5, 5, 1)
    anal_comp_alg.exponential_equation(5, 1, -10)
    anal_comp_alg.sin_x(4, -2, 2)
    anal_comp_alg.cos_x(7, 2, 3)
    anal_comp_alg.tg_x(-5, 3, -2)
    anal_comp_alg.ctg_x(4, 8, -10)
    anal_comp_alg.save(Config.PATH_TO_STATISTIC)


if __name__ == '__main__':
    run_statistic()
