#!/usr/bin/env python3

import numpy as np
from sklearn import neural_network
import time
# # https://stackoverflow.com/questions/61867945/python-import-error-cannot-import-name-six-from-sklearn-externals
# import six
# import sys
# sys.modules['sklearn.externals.six'] = six
# import mlrose

# from mlrose.algorithms import random_hill_climb as rhc
# from mlrose.algorithms import simulated_annealing as sa
# from mlrose.algorithms import genetic_alg as ga
# from mlrose.algorithms import mimic as mimic
# from mlrose.decay import ExpDecay
# from mlrose.neural import NeuralNetwork
# from mlrose.opt_probs import DiscreteOpt
# from mlrose.fitness import FourPeaks    # Better for Genetic Algorithms
# from mlrose.fitness import OneMax       # Better for Simulated Annealing & Randomized Hill Climibing
# from mlrose.fitness import Knapsack     # Better for MIMIC
# from mlrose.fitness import MaxKColor
# from mlrose.fitness import FlipFlop


# Acknowledgements - Referred to works from
# Rollings, A. (2020). mlrose: Machine Learning, Randomized Optimization and SEarch pack-age for Python,
# hiive extended remix. https://github.com/hiive/mlrose. Accessed: 14 March 2021.
# Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python.
# https://github.com/gkhayes/mlrose. Accessed: 14 March 2021.
# David S. Park for the MIMIC enhancements (from https://github.com/parkds/mlrose)
from mlrose_hiive.algorithms import random_hill_climb as rhc
from mlrose_hiive.algorithms import simulated_annealing as sa
from mlrose_hiive.algorithms import genetic_alg as ga
from mlrose_hiive.algorithms import mimic as mimic
from mlrose_hiive.algorithms.decay import ExpDecay
from mlrose_hiive.neural import NeuralNetwork
from mlrose_hiive.opt_probs import DiscreteOpt
from mlrose_hiive.fitness import FourPeaks    # Better for Genetic Algorithms
from mlrose_hiive.fitness import OneMax       # Better for Simulated Annealing & Randomized Hill Climibing
from mlrose_hiive.fitness import Knapsack     # Better for MIMIC
from mlrose_hiive.fitness import MaxKColor
from mlrose_hiive.fitness import FlipFlop


class RandomOptProblems(object):
    def __init__(self, fitness="FourPeaks", outfolder="./", attempts=1000):
        if fitness == "FourPeaks":
            self.fitness = FourPeaks(t_pct=0.1)
            self.problem = DiscreteOpt(length=100, fitness_fn=self.fitness, maximize=True, max_val=2)
        elif fitness == "Knapsack":
            self.fitness = Knapsack(weights=[10, 5, 2, 8, 15], values=[1, 2, 3, 4, 5], max_weight_pct=0.4)
            self.problem = DiscreteOpt(length=5, fitness_fn=self.fitness, maximize=True, max_val=2)
        elif fitness == "MaxKColor":
            self.fitness = MaxKColor([(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)])
            self.problem = DiscreteOpt(length=5, fitness_fn=self.fitness, maximize=True, max_val=2)
        elif fitness == "FlipFlop":
            self.fitness = FlipFlop()
            self.problem = DiscreteOpt(length=100, fitness_fn=self.fitness, maximize=True, max_val=2)
        self.outdir = outfolder
        self.attempts = attempts
        self.start_time = 0.0
        self.store_time = []
        # setup parameters for each optimizers if given

    def optimize_rhc(self,
                     rhc_iterations=10000):
        exec_time = []
        curves = []
        rhc_state, rhc_fitness, rhc_curves = rhc(problem=self.problem,
                                                 max_attempts=self.attempts,
                                                 max_iters=rhc_iterations,
                                                 restarts=0,
                                                 init_state=None,
                                                 curve=True,
                                                 random_state=None,
                                                 state_fitness_callback=self.execution_time_cb,
                                                 callback_user_info=[])
        # exec_time.append(self.store_time)
        # curves.append(rhc_curves)
        # return rhc_state, rhc_fitness, curves, exec_time
        return rhc_state, rhc_fitness, rhc_curves, self.store_time

    def optimize_sa(self,
                    sa_iterations=10000,
                    sa_init_temp=1.00,
                    sa_min_temp=0.001,
                    sa_decay_rate=0.005):
        # simulated annealing parameters
        exec_time = []
        curves = []
        decay_rate = ExpDecay(init_temp=sa_init_temp,
                              exp_const=sa_decay_rate,
                              min_temp=sa_min_temp)
        sa_state, sa_fitness, sa_curves = sa(problem=self.problem,
                                             schedule=decay_rate,
                                             max_attempts=self.attempts,
                                             max_iters=sa_iterations,
                                             init_state=None,
                                             curve=True,
                                             random_state=None,
                                             state_fitness_callback=self.execution_time_cb,
                                             callback_user_info=[])
        # exec_time.append(self.store_time)
        # curves.append(sa_curves)
        # return sa_state, sa_fitness, curves, exec_time
        return sa_state, sa_fitness, sa_curves, self.store_time

    def optimize_ga(self,
                    ga_iterations=250,
                    ga_pop_size=200,
                    ga_mut_prob=0.1,
                    ga_pop_breed_pct=0.75):
        # ga parameters
        exec_time = []
        curves = []
        ga_state, ga_fitness, ga_curves = ga(problem=self.problem,
                                             pop_size=ga_pop_size,
                                             mutation_prob=ga_mut_prob,
                                             pop_breed_percent=ga_pop_breed_pct,
                                             max_attempts=self.attempts,
                                             max_iters=ga_iterations,
                                             curve=True,
                                             random_state=None,
                                             state_fitness_callback=self.execution_time_cb,
                                             callback_user_info=[])
        # exec_time.append(self.store_time)
        # curves.append(ga_curves)
        # return ga_state, ga_fitness, curves, exec_time
        return ga_state, ga_fitness, ga_curves, self.store_time

    def optimize_mimic(self,
                       mimic_iterations=250,
                       mimic_pop_size=200,
                       mimic_keep_pct=0.2):
        # mimic parameters
        exec_time = []
        curves = []
        self.problem.set_mimic_fast_mode(fast_mode=True)
        mimic_state, mimic_fitness, mimic_curves = mimic(problem=self.problem,
                                                         pop_size=mimic_pop_size,
                                                         keep_pct=mimic_keep_pct,
                                                         max_attempts=self.attempts,
                                                         max_iters=mimic_iterations,
                                                         curve=True,
                                                         random_state=None,
                                                         state_fitness_callback=self.execution_time_cb,
                                                         callback_user_info=[])
        # exec_time.append(self.store_time)
        # curves.append(mimic_curves)
        # return mimic_state, mimic_fitness, curves, exec_time
        return mimic_state, mimic_fitness, mimic_curves, self.store_time

    def execution_time_cb(self,
                          iteration,
                          attempt=None,
                          state=None,
                          fitness=None,
                          fitness_evaluations=None,
                          user_data=None,
                          done=None,
                          curve=None
                          ):
        if iteration == 0:
            self.start_time = time.time()
            self.store_time = []
        else:
            self.store_time.append(time.time() - self.start_time)

        return True


if __name__ == "__main__":
    print(" ML Randomized Optimization")

    # rop = RandomOptProblems("FourPeaks",
    #                         outfolder="\\",
    #                         attempts=1000)
    #
    # best_state, best_fit, curves = rop.optimize_rhc(rhc_iterations=100000)
    #
    # print("best_sate=", best_state)
    # print("best_fit=", best_fit)
    # print("curves=", curves)
    #
    # decay_range = [0.002, 0.02, 0.1]
    # for decay in decay_range:
    #     rop = RandomOptProblems("FourPeaks",
    #                             outfolder="\\",
    #                             attempts=1000)
    #
    #     best_state, best_fit, curves = rop.optimize_sa(sa_iterations=100000,
    #                                                    sa_init_temp=100,
    #                                                    sa_min_temp=decay,
    #                                                    sa_decay_rate=0.001)
    #
    #     print("best_sate=", best_state)
    #     print("best_fit=", best_fit)
    #     print("curves=", curves)


