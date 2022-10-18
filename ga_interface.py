import pygad

from tools import benchmark


class GAInterface:
    def __init__(self, equation):
        self.equation = equation  # equation should = 0
        self.ga_instance = None

    def fitness_func(self, solution, solution_idx):
        x = sum(solution)  # x use in eval(equation)
        func_val = eval(self.equation)
        fitness_val = 1 / func_val + 0.00000000001
        return fitness_val

    def build_solver(self, **kwargs):
        stop_criteria = 'reach_' + str(int(1 / kwargs['accuracy']))
        self.ga_instance = pygad.GA(**kwargs,
                                    fitness_func=self.fitness_func,
                                    stop_criteria=stop_criteria)

    @benchmark
    def run_solver(self):
        self.ga_instance.run() if self.ga_instance else None

    def get_result(self):
        solution, sol_fitness, sol_idx = self.ga_instance.best_solution(self.ga_instance.last_generation_fitness)
        return {
            'x': sum(solution),
            'fitness': sol_fitness,
            'figure': self.ga_instance.plot_fitness()
        }
