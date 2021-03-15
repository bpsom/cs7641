#!/usr/bin/env python3
import os
import math
import sys
import getopt
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.neural_network import MLPClassifier
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

from sklearn import datasets
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
# scaling the data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, classification_report
from sklearn.model_selection import train_test_split


import RandomOptProblems as rop

win = True


def plot_optimization(title, label, xlabel, ylabel, xdata, ydata):
    plt.plot(xdata, ydata, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def optimization_performance(ropp, iter, prob_name, perf_name, savefold):
    rhc_iter = iter[0]
    sa_iter = iter[1]
    ga_iter = iter[2]
    mimic_iter = iter[3]

    fig = plt.figure()
    # Performance evaluation using default settings of optimizers
    print("--------------------RHC Performance---------------------")
    pbest_state, rhc_best_fit, rhc_curves, rhc_exec_times = ropp.optimize_rhc(rhc_iterations=rhc_iter)
    fp.write("--RHC Performance--" + "\n")
    fp.write("best_fit=" + str(rhc_best_fit) + "\n")
    fp.write("AverageTime=" + str(np.mean(rhc_exec_times)) + "\n")
    rtime = rhc_exec_times
    # rtime = rhc_exec_times[0]
    # a = np.array(rhc_curves)
    # b = a[0, :]
    b = rhc_curves
    plot_optimization(prob_name, "RHC", "Evaluations", perf_name, b[:, 1],  b[:, 0])

    print("\n")
    print("--------------------SA Performance---------------------")
    pbest_state, sa_best_fit, sa_curves, sa_exec_times = ropp.optimize_sa(sa_iterations=sa_iter)
    fp.write("--SA Performance--" + "\n")
    fp.write("best_fit=" + str(sa_best_fit) + "\n")
    fp.write("AverageTime=" + str(np.mean(sa_exec_times)) + "\n")
    stime = sa_exec_times
    # stime = sa_exec_times[0]
    # a = np.array(sa_curves)
    # b = a[0, :]
    b = sa_curves
    plot_optimization(prob_name, "SA", "Evaluations", perf_name,  b[:, 1],  b[:, 0])

    print("\n")
    print("--------------------GA Performance---------------------")
    pbest_state, ga_best_fit, ga_curves, ga_exec_times = ropp.optimize_ga(ga_iterations=ga_iter)
    fp.write("--GA Performance--" + "\n")
    fp.write("best_fit=" + str(ga_best_fit) + "\n")
    fp.write("AverageTime=" + str(np.mean(ga_exec_times)) + "\n")
    gtime = ga_exec_times
    # gtime = ga_exec_times[0]
    # a = np.array(ga_curves)
    # b = a[0, :]
    b = ga_curves
    plot_optimization(prob_name, "GA", "Evaluations", perf_name,  b[:, 1],  b[:, 0])

    print("\n")
    print("--------------------MIMIC Performance---------------------")
    pbest_state, m_best_fit, m_curves, m_exec_times = ropp.optimize_mimic(mimic_iterations=mimic_iter)
    fp.write("--MIMIC Performance--" + "\n")
    fp.write("best_fit=" + str(m_best_fit) + "\n")
    fp.write("AverageTime=" + str(np.mean(m_exec_times)) + "\n")
    mtime = m_exec_times
    # mtime = m_exec_times[0]
    # a = np.array(m_curves)
    # b = a[0, :]
    b = m_curves
    plot_optimization(prob_name, "MIMIC", "Evaluations", perf_name,  b[:, 1],  b[:, 0])

    plt.savefig(savefold + '{}-{}.png'.format(prob_name, perf_name))
    plt.close(fig)

    fig1 = plt.figure()
    iter = np.arange(1, len(rtime)+1)
    plot_optimization(prob_name, "RHC", "Iterations", "Time (Seconds)", np.array(iter), np.array(rtime))
    iter = np.arange(1, len(stime) + 1)
    plot_optimization(prob_name, "SA", "Iterations", "Time (Seconds)", np.array(iter), np.array(stime))
    iter = np.arange(1, len(gtime) + 1)
    plot_optimization(prob_name, "GA", "Iterations", "Time (Seconds)", np.array(iter), np.array(gtime))
    iter = np.arange(1, len(mtime) + 1)
    plot_optimization(prob_name, "MIMIC", "Iterations", "Time (Seconds)", np.array(iter), np.array(mtime))
    plt.savefig(savefold + '{}-Time.png'.format(prob_name))
    plt.close(fig1)


def usage():
    print("python SUTestLearners.py -s <os_type>")
    print("-s = Operating System win or linux. 0 - win, 1 - linux")
    print("Results of the experiments are saved (text file and plots)"
          "in results/, results/iris/, results/winequality folders")


if __name__ == "__main__":
    print("--- Randomized Optimization ---" + "\n" + "\n")
    pd.plotting.register_matplotlib_converters()
    sns.set()

    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:h', ['os_type=', 'help'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    os_typ = 0
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-s', '--sys'):
            os_typ = int(arg)
        else:
            usage()
            sys.exit(2)

    if os_typ == 0:
        win = True
    else:
        win = False

    folder = "result"
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        fname='./{}/Results.txt'.format(folder)
        os.remove(fname)
        # if win:
        #     os.remove("results\\Results.txt")
        # else:
        #     os.remove("results/Results.txt")
    except OSError:
        pass

    dirname = './{}/'.format(folder)
    fname = './{}/Results.txt'.format(folder)
    fp = open(fname, "a+")

    # if win:
    #     fp = open("results\\Results.txt", "a+")
    # else:
    #     fp = open("results/Results.txt", "a+")

    fp.write("--- Randomized Optimization ---" + "\n" + "\n")

    csv_file = './Data/winequality-white.csv'
    df = pd.read_csv(csv_file, sep=';')
    # if win:
    #     df = pd.read_csv("Data\\winequality-white.csv", sep=';')
    # else:
    #     df = pd.read_csv("Data/winequality-white.csv", sep=';')

    # df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    #                  sep=';')
    # print(df.head(10))
    X = df.drop('quality', axis=1)
    # y = df['quality']
    samples = df['quality']
    y = [1 if sample > 5 else 0 for sample in samples]
    # print(X.shape, y.shape)
    wtrain_x, wtest_x, wtrain_y, wtest_y = train_test_split(X, y, test_size=0.4, random_state=1)


    # Applying Standard scaling to get optimized result
    sc = StandardScaler()
    wtrain_x = sc.fit_transform(wtrain_x)
    wtest_x = sc.fit_transform(wtest_x)

    dataset = [(wtrain_x, wtest_x, wtrain_y, wtest_y)]

    # save_folder = []
    #
    # if win:
    #     save_folder.append("results\\winequality")
    # else:
    #     save_folder.append("results/winequality")

    # Max iterations
    rhc_iterations = 10000
    # simulated annealing parameters
    sa_iterations = 10000
    sa_init_temp = 100
    sa_min_temp = 0.001
    sa_decay_rates = [0.002, 0.02, 0.2]
    # ga parameters
    ga_iterations = 250
    ga_pop_size = [200, 500, 1000]
    ga_mut_prob = 0.1
    ga_pop_breed_pct = [0.75, 0.5, 0.25]
    # mimic parameters
    mimic_iterations = 250
    mimic_pop_size = [200, 500, 1000]
    mimic_keep_pct = [0.1, 0.2, 0.3]

    iterations = [rhc_iterations, sa_iterations, ga_iterations, mimic_iterations]

    # For each problem, call each optimizer,
    # record result - best function, best optimal value
    # and plot the results using the curves
    # Compare the results of optimizer for each problem

    ##################################################################
    # 1. Four Peaks problem
    ##################################################################
    rop_obj = rop.RandomOptProblems(fitness="FourPeaks",
                                    outfolder=folder,
                                    attempts=1000)

    print("\n")
    print("--------------------Four Peaks Evaluation---------------------")
    fp.write("\n" + "--------------------Four Peaks Evaluation---------------------" + "\n")
    prob_name = "Four Peaks"
    perf_name = "Fitness"
    # -------------------------------
    optimization_performance(rop_obj, iterations, prob_name, perf_name, dirname)
    # -------------------------------

    ##################################################################
    # 2. FlipFlop problem
    ##################################################################
    rop_obj = rop.RandomOptProblems(fitness="FlipFlop",
                                    outfolder=folder,
                                    attempts=1000)
    print("\n")
    print("--------------------FlipFlop Evaluation ---------------------")
    fp.write("\n" + "--------------------FlipFlop Evaluation---------------------" + "\n")
    prob_name = "FlipFlop"
    perf_name = "Fitness"
    # -------------------------------
    optimization_performance(rop_obj, iterations, prob_name, perf_name, dirname)
    # -------------------------------

    ##################################################################
    # 3. Knapsack problem
    ##################################################################
    rop_obj = rop.RandomOptProblems(fitness="Knapsack",
                                    outfolder=folder,
                                    attempts=1000)

    print("\n")
    print("--------------------Knapsack Evaluation ---------------------")
    fp.write("\n" + "--------------------Knapsack Evaluation---------------------" + "\n")
    prob_name = "Knapsack"
    perf_name = "Fitness"
    # -------------------------------
    optimization_performance(rop_obj, iterations, prob_name, perf_name, dirname)
    # -------------------------------


    ##################################################################
    # 4. MaxKColor problem
    ##################################################################
    rop_obj = rop.RandomOptProblems(fitness="MaxKColor",
                                    outfolder=folder,
                                    attempts=1000)

    print("\n")
    print("--------------------MaxKColor Evaluation ---------------------")
    fp.write("\n" + "--------------------MaxKColor Evaluation---------------------" + "\n")
    prob_name = "MaxKColor"
    perf_name = "Fitness"
    # -------------------------------
    optimization_performance(rop_obj, iterations, prob_name, perf_name, dirname)
    # -------------------------------

    # # After all iterations for both Datasets
    plt.close('all')
    fp.write("Done")
    fp.close()
    print("Done")





    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Hyper parameter tuning

    ##################################################################
    # 1. Four Peaks problem
    ##################################################################
    rop_obj = rop.RandomOptProblems(fitness="FourPeaks",
                                    outfolder=folder,
                                    attempts=1000)

    print("\n")
    print("--------------------Four Peaks Evaluation---------------------")
    prob_name = "Four Peaks"
    perf_name = "Fitness"

    fp.write("\n\n\n" + "--Hyper parameter tuning --" + "\n")

    # 1.2 call Simulated Annealing
    curve = []
    times = []
    for decay in sa_decay_rates:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_sa(sa_iterations=sa_iterations,
                                                                       sa_init_temp=sa_init_temp,
                                                                       sa_min_temp=sa_min_temp,
                                                                       sa_decay_rate=decay)
        fp.write("--SA Hyper parameter --" + "\n")
        fp.write("Decay=" + str(decay) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(sa_decay_rates, curve, label="SA")
    plt.title("SA Fitness - Tuning Temperature with exponential decay")
    plt.xlabel("Temperature Exponential Decay Rate")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-SA-Decay.png'.format(prob_name))
    # plt.show()
    plt.close(fig)


    # 1.3 call Genetic Algorithm
    # for pop_size in ga_pop_size:
    #     for pop_breed_pct in ga_pop_breed_pct:
    #         best_state, best_fit, curves, exec_times = rop_obj.optimize_ga(ga_iterations=ga_iterations,
    #                                                                        ga_pop_size=pop_size,
    #                                                                        ga_mut_prob=ga_mut_prob,
    #                                                                        ga_pop_breed_pct=pop_breed_pct)
    #         fp.write("--GA Hyper parameter --" + "\n")
    #         fp.write("Population Size=" + str(pop_size) + "\n")
    #         fp.write("Population Breed %=" + str(pop_breed_pct) + "\n")
    #         fp.write("best_fit=" + str(best_fit) + "\n")
    #         fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
    curve = []
    times = []
    for pop_size in ga_pop_size:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_ga(ga_iterations=ga_iterations,
                                                                       ga_pop_size=pop_size,
                                                                       ga_mut_prob=ga_mut_prob,
                                                                       ga_pop_breed_pct=0.5)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(pop_size) + "\n")
        fp.write("Population Breed %=" + str(0.5) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(ga_pop_size, curve, label="GA")
    plt.title("GA Fitness - population size")
    plt.xlabel("Population Size")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-GA-popsize.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    curve = []
    times = []
    for pop_breed_pct in ga_pop_breed_pct:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_ga(ga_iterations=ga_iterations,
                                                                       ga_pop_size=200,
                                                                       ga_mut_prob=ga_mut_prob,
                                                                       ga_pop_breed_pct=pop_breed_pct)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(200) + "\n")
        fp.write("Population Breed ratio=" + str(pop_breed_pct) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(ga_pop_breed_pct, curve, label="GA")
    plt.title("GA Fitness - population breeding ratio")
    plt.xlabel("Population breeding ratio")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-GGA-popbreed.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # 1.4 call MIMIC
    curve = []
    times = []
    for pop_size in mimic_pop_size:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_mimic(mimic_iterations=mimic_iterations,
                                                                          mimic_pop_size=pop_size,
                                                                          mimic_keep_pct=0.2)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(pop_size) + "\n")
        fp.write("Keep population %=" + str(0.2*100) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(mimic_pop_size, curve, label="MIMIC")
    plt.title("MIMIC Fitness - population size")
    plt.xlabel("Population size")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-MIMIC-popsize.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    curve = []
    times = []
    for keep_pct in mimic_keep_pct:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_mimic(mimic_iterations=mimic_iterations,
                                                                          mimic_pop_size=200,
                                                                          mimic_keep_pct=keep_pct)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(200) + "\n")
        fp.write("Keep population %=" + str(keep_pct * 100) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(mimic_keep_pct, curve, label="MIMIC")
    plt.title("MIMIC Fitness - population keep ratio")
    plt.xlabel("Population keeping ratio")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-MIMIC-popkeep.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("done - FourPeaks")
    print("##################################################################")

    ##################################################################
    # 2. FlipFlop problem
    ##################################################################
    rop_obj = rop.RandomOptProblems(fitness="FlipFlop",
                                    outfolder=folder,
                                    attempts=1000)
    print("\n")
    print("--------------------FlipFlop Evaluation ---------------------")
    prob_name = "FlipFlop"
    perf_name = "Fitness"

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Hyper parameter tuning
    # 2.1 call Random Hill Climbing

    # 2.2 call Simulated Annealing
    curve = []
    times = []
    for decay in sa_decay_rates:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_sa(sa_iterations=sa_iterations,
                                                                       sa_init_temp=sa_init_temp,
                                                                       sa_min_temp=sa_min_temp,
                                                                       sa_decay_rate=decay)
        fp.write("--SA Hyper parameter --" + "\n")
        fp.write("Decay=" + str(decay) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(sa_decay_rates, curve, label="SA")
    plt.title("SA Fitness - Tuning Temperature with exponential decay")
    plt.xlabel("Temperature Exponential Decay Rate")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-SA-Decay.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # 2.3 call Genetic Algorithm
    curve = []
    times = []
    for pop_size in ga_pop_size:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_ga(ga_iterations=ga_iterations,
                                                                       ga_pop_size=pop_size,
                                                                       ga_mut_prob=ga_mut_prob,
                                                                       ga_pop_breed_pct=0.5)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(pop_size) + "\n")
        fp.write("Population Breed %=" + str(0.5) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(ga_pop_size, curve, label="GA")
    plt.title("GA Fitness - population size")
    plt.xlabel("Population Size")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-GA-popsize.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    curve = []
    times = []
    for pop_breed_pct in ga_pop_breed_pct:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_ga(ga_iterations=ga_iterations,
                                                                       ga_pop_size=200,
                                                                       ga_mut_prob=ga_mut_prob,
                                                                       ga_pop_breed_pct=pop_breed_pct)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(200) + "\n")
        fp.write("Population Breed %=" + str(pop_breed_pct) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(ga_pop_breed_pct, curve, label="GA")
    plt.title("GA Fitness - population breeding ratio")
    plt.xlabel("Population breeding ratio")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-GA-popbreed.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # 2.4 call MIMIC
    curve = []
    times = []
    for pop_size in mimic_pop_size:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_mimic(mimic_iterations=mimic_iterations,
                                                                          mimic_pop_size=pop_size,
                                                                          mimic_keep_pct=0.2)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(pop_size) + "\n")
        fp.write("Keep population %=" + str(0.2 * 100) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(mimic_pop_size, curve, label="MIMIC")
    plt.title("MIMIC Fitness - population size")
    plt.xlabel("Population size")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-MIMIC-popsize.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    curve = []
    times = []
    for keep_pct in mimic_keep_pct:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_mimic(mimic_iterations=mimic_iterations,
                                                                          mimic_pop_size=200,
                                                                          mimic_keep_pct=keep_pct)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(200) + "\n")
        fp.write("Keep population %=" + str(keep_pct * 100) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(mimic_keep_pct, curve, label="MIMIC")
    plt.title("MIMIC Fitness - population keep ratio")
    plt.xlabel("Population keeping ratio")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-MIMIC-popkeep.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("done - FlipFlop")
    print("##################################################################")

    ##################################################################
    # 3. Knapsack problem
    ##################################################################
    rop_obj = rop.RandomOptProblems(fitness="Knapsack",
                                    outfolder=folder,
                                    attempts=1000)

    print("\n")
    print("--------------------Knapsack Evaluation ---------------------")
    prob_name = "Knapsack"
    perf_name = "Fitness"


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Hyper parameter tuning

    # 3.2 call Simulated Annealing
    curve = []
    times = []
    for decay in sa_decay_rates:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_sa(sa_iterations=sa_iterations,
                                                                       sa_init_temp=sa_init_temp,
                                                                       sa_min_temp=sa_min_temp,
                                                                       sa_decay_rate=decay)
        fp.write("--SA Hyper parameter --" + "\n")
        fp.write("Decay=" + str(decay) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(sa_decay_rates, curve, label="SA")
    plt.title("SA Fitness - Tuning Temperature with exponential decay")
    plt.xlabel("Temperature Exponential Decay Rate")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-SA-Decay.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # 3.3 call Genetic Algorithm
    curve = []
    times = []
    for pop_size in ga_pop_size:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_ga(ga_iterations=ga_iterations,
                                                                       ga_pop_size=pop_size,
                                                                       ga_mut_prob=ga_mut_prob,
                                                                       ga_pop_breed_pct=0.5)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(pop_size) + "\n")
        fp.write("Population Breed %=" + str(0.5) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(ga_pop_size, curve, label="GA")
    plt.title("GA Fitness - population size")
    plt.xlabel("Population Size")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-GA-popsize.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    curve = []
    times = []
    for pop_breed_pct in ga_pop_breed_pct:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_ga(ga_iterations=ga_iterations,
                                                                       ga_pop_size=200,
                                                                       ga_mut_prob=ga_mut_prob,
                                                                       ga_pop_breed_pct=pop_breed_pct)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(200) + "\n")
        fp.write("Population Breed %=" + str(pop_breed_pct) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(ga_pop_breed_pct, curve, label="GA")
    plt.title("GA Fitness - population breeding ratio")
    plt.xlabel("Population breeding ratio")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-GA-popbreed.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # 3.4 call MIMIC
    curve = []
    times = []
    for pop_size in mimic_pop_size:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_mimic(mimic_iterations=mimic_iterations,
                                                                          mimic_pop_size=pop_size,
                                                                          mimic_keep_pct=0.2)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(pop_size) + "\n")
        fp.write("Keep population %=" + str(0.2 * 100) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(mimic_pop_size, curve, label="MIMIC")
    plt.title("MIMIC Fitness - population size")
    plt.xlabel("Population size")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-MIMIC-popsize.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    curve = []
    times = []
    for keep_pct in mimic_keep_pct:
        best_state, best_fit, curves, exec_times = rop_obj.optimize_mimic(mimic_iterations=mimic_iterations,
                                                                          mimic_pop_size=200,
                                                                          mimic_keep_pct=keep_pct)
        fp.write("--GA Hyper parameter --" + "\n")
        fp.write("Population Size=" + str(200) + "\n")
        fp.write("Keep population %=" + str(keep_pct * 100) + "\n")
        fp.write("best_fit=" + str(best_fit) + "\n")
        fp.write("AverageTime=" + str(np.mean(exec_times)) + "\n")
        curve.append(best_fit)
        times.append(exec_times)
    fig = plt.figure()
    plt.plot(mimic_keep_pct, curve, label="MIMIC")
    plt.title("MIMIC Fitness - population keep ratio")
    plt.xlabel("Population keeping ratio")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.savefig(dirname + '{}-MIMIC-popkeep.png'.format(prob_name))
    # plt.show()
    plt.close(fig)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("done - Knapsack")
    print("##################################################################")





    print("##################################################################")
    print("Neural network weight optimization")

    # wtrain_x, wtest_x, wtrain_y, wtest_y
    episodes = [100, 500, 1000]
    # For sa
    exp_decay = ExpDecay(init_temp=100,
                         exp_const=0.02,
                         min_temp=0.001)

    plt.figure()
    train_val_figure = plt.gcf().number
    plt.figure()
    train_time_figure = plt.gcf().number

    x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(wtrain_x,
                                                                          wtrain_y,
                                                                          test_size=0.2,
                                                                          shuffle=True,
                                                                          stratify=wtrain_y)

    train_plot = []
    val_plot = []
    train_time_plot = []
    for episode in episodes:
        # Define Neural Network using current algorithm
        nn = NeuralNetwork(hidden_nodes=[30, 50], activation='relu',
                           algorithm='random_hill_climb', max_iters=int(episode),
                           bias=True, is_classifier=True,
                           early_stopping=False, learning_rate=0.001,
                           clip_max=1e10, schedule=exp_decay,
                           pop_size=200, mutation_prob=0.1,
                           max_attempts=1000, curve=False)

        # Train on current training fold and append training time
        start_time = time.time()
        nn.fit(x_train_fold, y_train_fold)
        # nn.fit(wtrain_x, wtrain_y)
        train_time_plot.append(time.time() - start_time)

        # Compute and append training and validation log losses
        tr_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
        train_plot.append(tr_loss)

        va_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
        val_plot.append(va_loss)

    # Plot
    plt.figure(train_val_figure)
    plt.plot(episodes, np.array(train_plot), label="RHC-Training")
    plt.plot(episodes, np.array(val_plot), label="RHC-Values")
    plt.figure(train_time_figure)
    plt.plot(episodes, np.array(train_time_plot), label="RHC-Timing")

    train_plot = []
    val_plot = []
    train_time_plot = []
    for episode in episodes:
        # Define Neural Network using current algorithm
        nn = NeuralNetwork(hidden_nodes=[30, 50], activation='relu',
                           algorithm='simulated_annealing', max_iters=int(episode),
                           bias=True, is_classifier=True,
                           early_stopping=False, learning_rate=0.001,
                           clip_max=1e10, schedule=exp_decay,
                           pop_size=200, mutation_prob=0.1,
                           max_attempts=1000, curve=False)

        # Train on current training fold and append training time
        start_time = time.time()
        nn.fit(x_train_fold, y_train_fold)
        train_time_plot.append(time.time() - start_time)

        # Compute and append training and validation log losses
        tr_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
        train_plot.append(tr_loss)

        va_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
        val_plot.append(va_loss)

    # Plot
    plt.figure(train_val_figure)
    plt.plot(episodes, np.array(train_plot), label="SA-Training")
    plt.plot(episodes, np.array(val_plot), label="SA-Values")
    plt.figure(train_time_figure)
    plt.plot(episodes, np.array(train_time_plot), label="SA-Timing")

    train_plot = []
    val_plot = []
    train_time_plot = []
    for episode in episodes:
        # Define Neural Network using current algorithm
        nn = NeuralNetwork(hidden_nodes=[30, 50], activation='relu',
                           algorithm='genetic_alg', max_iters=int(episode),
                           bias=True, is_classifier=True,
                           early_stopping=False, learning_rate=0.001,
                           clip_max=1e10, schedule=exp_decay,
                           pop_size=200, mutation_prob=0.1,
                           max_attempts=1000, curve=False)

        # Train on current training fold and append training time
        start_time = time.time()
        nn.fit(x_train_fold, y_train_fold)
        train_time_plot.append(time.time() - start_time)

        # Compute and append training and validation log losses
        tr_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
        train_plot.append(tr_loss)

        va_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
        val_plot.append(va_loss)

    # Plot
    plt.figure(train_val_figure)
    plt.plot(episodes, np.array(train_plot), label="GA-Training")
    plt.plot(episodes, np.array(val_plot), label="GA-Values")
    plt.figure(train_time_figure)
    plt.plot(episodes, np.array(train_time_plot), label="GA-Timing")
    #
    # train_plot = []
    # val_plot = []
    # train_time_plot = []
    # for episode in episodes:
    #     # Define Neural Network using current algorithm
    #     nn = NeuralNetwork(hidden_nodes=[30, 50], activation='relu',
    #                        algorithm='gradient_descent', max_iters=int(episode),
    #                        bias=True, is_classifier=True,
    #                        early_stopping=False, learning_rate=0.001,
    #                        clip_max=1e10, schedule=exp_decay,
    #                        pop_size=200, mutation_prob=0.1,
    #                        max_attempts=1000, curve=False)
    #
    #     # Train on current training fold and append training time
    #     start_time = time.time()
    #     nn.fit(x_train_fold, y_train_fold)
    #     train_time_plot.append(time.time() - start_time)
    #
    #     # Compute and append training and validation log losses
    #     tr_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
    #     train_plot.append(tr_loss)
    #
    #     va_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
    #     val_plot.append(va_loss)
    #
    # # Plot
    # plt.figure(train_val_figure)
    # plt.plot(episodes, np.array(train_plot), label="GD-Training")
    # plt.plot(episodes, np.array(val_plot), label="GD-Values")

    plt.title("Neural Network - Loss vs Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dirname + 'nn-episodes-loss.png')
    # plt.show()

    # plt.figure(train_time_figure)
    # plt.plot(episodes, np.array(train_time_plot), label="GD-Timing")
    plt.title("Neural Network - Time vs Episodes")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dirname + 'nn-time-loss.png')
    # plt.show()

    # Test Neural network
    # Define Neural Network using RHC for weights optimization
    nn = NeuralNetwork(hidden_nodes=[30, 50], activation='relu',
                       algorithm='random_hill_climb', max_iters=int(episode),
                       bias=True, is_classifier=True,
                       early_stopping=False, learning_rate=0.001,
                       clip_max=1e10, schedule=exp_decay,
                       pop_size=ga_pop_size, mutation_prob=ga_mut_prob,
                       max_attempts=1000, curve=False)

    nn_rhc = NeuralNetwork(hidden_nodes=[30, 50], activation='relu',
                           algorithm='random_hill_climb', max_iters=200,
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=False, clip_max=1e10,
                           max_attempts=200, curve=False)

    # Define Neural Network using SA for weights optimization
    nn_sa = NeuralNetwork(hidden_nodes=[30, 50], activation='relu',
                          algorithm='simulated_annealing', max_iters=200,
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10, schedule=exp_decay,
                          max_attempts=200, curve=False)

    # Define Neural Network using GA for weights optimization
    nn_ga = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
                          algorithm='genetic_alg', max_iters=200,
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10,
                          pop_size=200, mutation_prob=0.1,
                          max_attempts=200, curve=False)

    # # Define Neural Network using GD for weights optimization
    # nn_gd = NeuralNetwork(hidden_nodes=[50, 30], activation='relu',
    #                       algorithm='gradient_descent', max_iters=200,
    #                       bias=True, is_classifier=True, learning_rate=0.001,
    #                       early_stopping=False, clip_max=1e10,
    #                       max_attempts=200, curve=False)

    # Test the optimizartion by predicting
    # wtrain_x, wtest_x, wtrain_y, wtest_y

    # Print classification reports for all of the optimization algorithms
    nn_rhc.fit(wtrain_x, wtrain_y)
    fp.write('RHC classification report = \n {}'.format(classification_report(wtest_y, nn_rhc.predict(wtest_x))))
    nn_sa.fit(wtrain_x, wtrain_y)
    fp.write('SA classification report = \n {}'.format(classification_report(wtest_y, nn_sa.predict(wtest_x))))
    nn_ga.fit(wtrain_x, wtrain_y)
    fp.write('GA classification report = \n {}'.format(classification_report(wtest_y, nn_ga.predict(wtest_x))))
    # nn_gd.fit(wtrain_x, wtrain_y)
    # fp.write('GD classification report = \n {}'.format(classification_report(wtest_y, nn_gd.predict(wtest_x))))


    # # After all iterations for both Datasets
    plt.close('all')
    fp.write("Done")
    fp.close()
    print("Done")




