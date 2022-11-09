import json

from config import Config
from ga_interface import GAInterface


class Analyser:
    def __init__(self, path_to_result, num_loop):
        self.path_to_result = path_to_result
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
        equation = f'{a}*(x**{b})+{c}'
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

    def save(self):
        with open(self.path_to_result+'result.json', 'w') as file:
            file.write(json.dumps(self.result, indent=4))

    def _run(self, ga):
        execution_times, ga_result = [], {}
        fails = 0
        for _ in range(self.num_loop):
            execution_time = ga.run_solver()
            ga_result = ga.get_result()
            if ga_result.get('error') > ga_result.get('accuracy'):
                fails += 1
                continue
            execution_times.append(execution_time)

        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        return {
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'attempts': self.num_loop,
            'fails': fails,
            'percent_fails': 100 * fails / self.num_loop,
            **ga_result
        }


def run_statistic():
    analyser = Analyser(Config.PATH_TO_STATISTIC, Config.NUM_LOOPS)
    analyser.linear_equation(5, -3)
    analyser.polynomial_2(2, 5, -15)
    analyser.polynomial_3(4, 2, 7, -5)
    analyser.polynomial_4(7, -2, -6, -3, 10)
    analyser.polynomial_5(-3, -4, -7, 5, 5, 1)
    analyser.exponential_equation(4, 3, 2)
    analyser.sin_x(4, 2, 2)
    analyser.cos_x(7, 1, 1)
    analyser.tg_x(-5, 3, -2)
    analyser.ctg_x(1, 1, 1)
    analyser.save()


if __name__ == '__main__':
    run_statistic()
