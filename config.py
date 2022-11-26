import os


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'SECRET_KEY')
    NUM_GENERATIONS = 1000
    NUM_PARENTS_MATING = 2
    NUM_GENES = 5
    SOL_PER_POP = 10
    CROSSOVER_TYPE = 'two_points'
    MUTATION_PROBABILITY = 0.15
    PARALLEL_PROCESSING = 1
    ACCURACY = 0.01
    ATTEMPTS = 10
    PATH_TO_STATISTIC_DATA = 'statistics/data/'
    PATH_TO_STATISTIC_IMG = 'statistics/images/'
