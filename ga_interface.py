import pygad, math

from tools import benchmark


class GAInterface:
    def __init__(self, equation):
        self.equation = equation  # equation should = 0
        self.ga_instance = None
        self.kwargs = None

    def build_solver(self, **kwargs):
        self.kwargs = kwargs

        def fitness_func(solution, solution_idx):
            x = sum(solution)  # x use in eval(equation)
            func_val = eval(self.equation)
            fitness_val = 1 / func_val + 0.00000000001
            return fitness_val

        accuracy = kwargs.pop('accuracy')
        stop_criteria = 'reach_' + str(int(1 / accuracy))
        self.ga_instance = pygad.GA(**kwargs,
                                    fitness_func=fitness_func,
                                    stop_criteria=stop_criteria)
        self.kwargs['accuracy'] = accuracy

    @benchmark
    def run_solver(self):
        self.ga_instance.run() if self.ga_instance else None

    def get_result(self):
        solution, sol_fitness, sol_idx = self.ga_instance.best_solution(self.ga_instance.last_generation_fitness)
        x = sum(solution)
        return {
            'x': x,
            'fitness': sol_fitness,
            'figure': self.ga_instance.plot_fitness(),
            'equation': self.equation,
            'error': eval(self.equation),
            **self.kwargs
        }
