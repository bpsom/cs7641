#!/usr/bin/env python3
import os
import math
import sys
import getopt
import numpy as np

import SUDecisionTree as sdt
import SUBoosting as sbt
import SUKnn as skn
import SUSvm as svmc
import SUNeuralNetworks as snn
# import SUBagging as sbg

import pandas as pd
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

import matplotlib.pyplot as plt
import time


from sklearn import svm
from sklearn.datasets import make_blobs

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

win = True


def plot_data(save_fold, plot_pfix, model_name, x_name, x_value, score_train, score_test, confused):
    plt.figure()
    plt.title(plot_pfix + "-" + model_name + "-tuning-" + x_name)
    plt.xlabel(x_name)
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    plt.grid()
    # plot the average training and test score lines at each training set size
    # if not model_name == "SupportVectorClassifier":
    plt.plot(x_value, score_train, 'o-', color="r")
    plt.plot(x_value, score_test, 'o-', color="g")
    # else:
    #     plt.plot(score_train, 'o-', color="r")
    #     plt.plot(score_test, 'o-', color="g")
    plt.legend(["Training score", "Cross-validation score"])
    # shows scores from 0 to 1.1
    plt.ylim(-.1, 1.1)
    global win
    if win:
        filename = save_fold + '\\' + '{}-{}-{}-tuning.png'.format(plot_pfix, model_name, x_name)
        print("Plot "+ filename + " Created")
        fp.write("Plot "+ filename + " Created" + "\n")
        plt.savefig(save_fold + '\\' + '{}-{}-{}-tuning.png'.format(plot_pfix, model_name, x_name))
    else:
        filename = save_fold + '/' + '{}-{}-{}-tuning.png'.format(plot_pfix, model_name, x_name)
        print("Plot " + filename + " Created")
        fp.write("Plot " + filename + " Created" + "\n")
        plt.savefig(save_fold + '/' + '{}-{}-{}-tuning.png'.format(plot_pfix, model_name, x_name))
    # if model_name == "SupportVectorClassifier":
    #     plt.show()
    plt.close()


def accuracy_test(fpt, y_train, train_pred_y, y_test, test_pred_y):
    train_score = metrics.accuracy_score(y_train, train_pred_y)
    test_score = metrics.accuracy_score(y_test, test_pred_y)
    confu_matrix = metrics.confusion_matrix(y_test, test_pred_y)
    # print("Train Data Prediction Accuracy %: ", (train_score * 100))
    # print("Test Data Validation Accuracy %: ", (test_score * 100))
    # print("Test Data Confusion Matrix :", confu_matrix)
    if fpt is not None:
        fpt.write("Train Data Prediction Accuracy %: ")
        fpt.write("".join(str(train_score * 100)) + "\n")
        fpt.write("Test Data Validation Accuracy %: ")
        fpt.write("".join(str(test_score * 100)) + "\n")
        fpt.write("Test Data Confusion Matrix :" + "\n")
        for line in confu_matrix:
            fp.write("".join(str(line)) + "\n")
    return train_score, test_score, confu_matrix


def plot_learning_curve(su_model, save_fold, plot_pfix, su_estimator, xx_train, yy_train):
    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator=su_estimator,
                                                                            X=xx_train,
                                                                            y=yy_train,
                                                                            train_sizes=np.linspace(0.1, 1.0, 5),
                                                                            cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title(plot_pfix + "-" +  su_model + " Learning Curve")
    plt.xlabel("train set size")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    plt.grid()
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g")
    plt.legend(["Training score", "Cross-validation score"])
    # shows scores from 0 to 1.1
    plt.ylim(-.1, 1.1)
    global win
    if win:
        filename = save_fold + '\\' + '{}-{}-LearningCurve.png'.format(plot_pfix, su_model)
        print("Plot " + filename + " Created")
        fp.write("Plot " + filename + " Created" + "\n")
        plt.savefig(save_fold + '\\' + '{}-{}-LearningCurve.png'.format(plot_pfix, su_model))
    else:
        filename = save_fold + '/' + '{}-{}-LearningCurve.png'.format(plot_pfix, su_model)
        print("Plot " + filename + " Created")
        fp.write("Plot " + filename + " Created" + "\n")
        plt.savefig(save_fold + '/' + '{}-{}-LearningCurve.png'.format(plot_pfix, su_model))
    # plt.show()
    plt.close()


def usage():
    print("python SUTestLearners.py -s <os_type>")
    print("-s = Operating System win or linux. 0 - win, 1 - linux")
    print("Results of the experiments are saved (text file and plots)"
          "in results/, results/iris/, results/winequality folders")


if __name__ == "__main__":
    print("--- Supervised Learning ---" + "\n" + "\n")

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

    try:
        if win:
            os.remove("results\\Results.txt")
        else:
            os.remove("results/Results.txt")
    except OSError:
        pass

    if win:
        fp = open("results\\Results.txt", "a+")
    else:
        fp = open("results/Results.txt", "a+")
    fp.write("--- Supervised Learning ---" + "\n" + "\n")

    iris = datasets.load_iris()
    itrain_x, itest_x, itrain_y, itest_y = train_test_split(iris.data, iris.target, test_size=0.4, random_state=7)
    # Applying Standard scaling to get optimized result
    sc = StandardScaler()
    itrain_x = sc.fit_transform(itrain_x)
    itest_x = sc.fit_transform(itest_x)

    if win:
        df = pd.read_csv("Data\\winequality-white.csv", sep=';')
    else:
        df = pd.read_csv("Data/winequality-white.csv", sep=';')
    # df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    #                  sep=';')
    # print(df.head(10))
    X = df.drop('quality', axis=1)
    y = df['quality']
    # print(X.shape, y.shape)
    wtrain_x, wtest_x, wtrain_y, wtest_y = train_test_split(X, y, test_size=0.4, random_state=1)

    # Applying Standard scaling to get optimized result
    sc = StandardScaler()
    wtrain_x = sc.fit_transform(wtrain_x)
    wtest_x = sc.fit_transform(wtest_x)
    # Statistical characteristics of each numerical feature
    # print(df.describe())
    # corr_matrix = df.corr()
    # print(corr_matrix["quality"].sort_values(ascending=False))

    dataset = [(itrain_x, itest_x, itrain_y, itest_y), (wtrain_x, wtest_x, wtrain_y, wtest_y)]
    plot_prefix = ["iris", "wine-Q"]
    save_folder = []
    if win:
        save_folder.append("results\\iris")
        save_folder.append("results\\winequality")
    else:
        save_folder.append("results/iris")
        save_folder.append("results/winequality")

    # dataset = [(wtrain_x, wtest_x, wtrain_y, wtest_y)]
    # plot_prefix = ["wine-Q"]
    # save_folder = []
    # if win:
    #     # save_folder.append("results\\iris")
    #     save_folder.append("results\\winequality")
    # else:
    #     # save_folder.append("results/iris")
    #     save_folder.append("results/winequality")

    for idx, s_data in enumerate(dataset):

        fp.write("Results for Data set =")
        fp.write(" ".join(plot_prefix[idx]) + "\n")

        print("Decision Tree Classifier")
        fp.write("\n" +"\n" + "Decision Tree Classifier" + "\n")

        train_x, test_x, train_y, test_y = s_data[0], s_data[1], s_data[2], s_data[3]
        models = []

        # 1. Decision Tree
        features = train_x.shape[1]
        # create a learner and train it
        learner = sdt.SUDecisionTree(leaf_size=1, features=features, modeltype=0)
        learner.add_evidence(train_x, train_y)  # train it
        mod = ('DecisionTreeClassifier', DecisionTreeClassifier())
        models.append(mod)
        plot_learning_curve(su_model=mod[0],
                            save_fold=save_folder[idx],
                            plot_pfix=plot_prefix[idx],
                            su_estimator=mod[1],
                            xx_train=train_x,
                            yy_train=train_y)

        # Predict for test data
        pred_y = learner.model_prediction(test_x)  # get the predictions
        pred_train_y = learner.model_prediction(train_x)  # get the predictions
        accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)

        # Hyperparameter Tuning
        # Tree Pre-pruning 1
        # feature selection, leaf_size
        features = train_x.shape[1]
        # if features >= 3:
        #     features = features - 1
        tr_scores = []
        te_scores = []
        c_matrixs = []
        x_value = range(1, features, 1)
        for k in x_value:
            # create a learner and train it
            learner = sdt.SUDecisionTree(leaf_size=1, features=k, modeltype=0)
            learner.add_evidence(train_x, train_y)  # train it
            # Predict for test data
            pred_y = learner.model_prediction(test_x)  # get the predictions
            pred_train_y = learner.model_prediction(train_x)  # get the predictions
            tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
            tr_scores.append(tr_score)
            te_scores.append(te_score)
            c_matrixs.append(c_matrix)
        plot_data(save_folder[idx], plot_prefix[idx], mod[0], "features", x_value, tr_scores, te_scores, c_matrixs)

        # Tree Pre-pruning 2
        # leaf_size
        tr_scores = []
        te_scores = []
        c_matrixs = []
        features = train_x.shape[1]
        x_value = range(1, train_x.shape[1], 1)
        for k in x_value:
            # create a learner and train it
            learner = sdt.SUDecisionTree(leaf_size=k, features=features, modeltype=0)
            learner.add_evidence(train_x, train_y)  # train it
            # Predict for test data
            pred_y = learner.model_prediction(test_x)  # get the predictions
            pred_train_y = learner.model_prediction(train_x)  # get the predictions
            tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
            tr_scores.append(tr_score)
            te_scores.append(te_score)
            c_matrixs.append(c_matrix)
        plot_data(save_folder[idx], plot_prefix[idx], mod[0], "leaf_size", x_value, tr_scores, te_scores, c_matrixs)

        # 2. Boosting - Adaboost
        print("Boosting - AdaBoost")
        fp.write("\n" +"\n" + "Boosting - AdaBoost" + "\n")
        features = train_x.shape[1]
        # create a learner and train it
        learner = sbt.SUBoosting(leaf_size=2, features=features, modeltype=0)
        learner.add_evidence(train_x, train_y)  # train it

        mod = ("AdaBoostClassifier", AdaBoostClassifier())
        models.append(mod)
        plot_learning_curve(su_model=mod[0],
                            save_fold=save_folder[idx],
                            plot_pfix=plot_prefix[idx],
                            su_estimator=mod[1],
                            xx_train=train_x,
                            yy_train=train_y)
        # Predict for test data
        pred_y = learner.model_prediction(test_x)  # get the predictions
        pred_train_y = learner.model_prediction(train_x)  # get the predictions
        tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)

        # Tree Pre-pruning 1
        # feature selection, leaf_size=2
        tr_scores = []
        te_scores = []
        c_matrixs = []
        features = train_x.shape[1]
        x_value = range(1, features, 1)
        for k in x_value:
            # create a learner and train it
            learner = sbt.SUBoosting(leaf_size=2, features=features, modeltype=0)
            learner.add_evidence(train_x, train_y)  # train it
            # Predict for test data
            pred_y = learner.model_prediction(test_x)  # get the predictions
            pred_train_y = learner.model_prediction(train_x)  # get the predictions
            tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
            tr_scores.append(tr_score)
            te_scores.append(te_score)
            c_matrixs.append(c_matrix)
        plot_data(save_folder[idx], plot_prefix[idx], mod[0], "features", x_value, tr_scores, te_scores, c_matrixs)

        # Tree Pre-pruning 2
        # leaf_size, features-2
        tr_scores = []
        te_scores = []
        c_matrixs = []
        features = train_x.shape[1]
        x_value = range(2, train_x.shape[1], 1)
        for k in x_value:
            # create a learner and train it
            learner = sbt.SUBoosting(leaf_size=k, features=features-2, modeltype=0)
            learner.add_evidence(train_x, train_y)  # train it
            # Predict for test data
            pred_y = learner.model_prediction(test_x)  # get the predictions
            pred_train_y = learner.model_prediction(train_x)  # get the predictions
            tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
            tr_scores.append(tr_score)
            te_scores.append(te_score)
            c_matrixs.append(c_matrix)
        plot_data(save_folder[idx], plot_prefix[idx], mod[0], "leaf_size", x_value, tr_scores, te_scores, c_matrixs)

        # 3. KNN
        print("K-Nearest Neighbors")
        fp.write("\n" +"\n" + "K-Nearest Neighbors" + "\n")
        k_range = range(1, 26)
        tr_scores = []
        te_scores = []
        c_matrixs = []
        saved = False
        for k in k_range:
            # create a learner and train it
            learner = skn.SUKnn(k)
            learner.add_evidence(train_x, train_y)  # train it
            if not saved:
                mod = ("KNeighborsClassifier", KNeighborsClassifier())
                models.append(mod)
                plot_learning_curve(su_model=mod[0],
                                    save_fold=save_folder[idx],
                                    plot_pfix=plot_prefix[idx],
                                    su_estimator=mod[1],
                                    xx_train=train_x,
                                    yy_train=train_y)
                saved = True
            # Predict for test data
            pred_y = learner.model_prediction(test_x)  # get the predictions
            pred_train_y = learner.model_prediction(train_x)  # get the predictions
            tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
            tr_scores.append(tr_score)
            te_scores.append(te_score)
            c_matrixs.append(c_matrix)
        plot_data(save_folder[idx], plot_prefix[idx], mod[0], "K-neighbors", k_range, tr_scores, te_scores, c_matrixs)

        # plt.plot(k_range, scores_list)
        # plt.xlabel("k value")
        # plt.ylabel("Model Accuracy - Test Data")
        # plt.title("KNN model accuracy: " + plot_prefix[idx])
        # plt.show()

        # 4. SVM
        print("Support Vector Machine")
        fp.write("\n" +"\n" + "Support Vector Machine" + "\n")
        # pipeline = Pipeline([('vect', CountVectorizer()),
        #                      ('tfidf', TfidfTransformer()),
        #                      ('clf', SVC())])
        #
        # h_params = {
        #     'clf__C': [0.8, 0.9, 1, 1.1, 1.2],
        #     'clf__gamma': [0.8, 0.9, 1, 1.1, 1.2],
        #     'clf__kernel': ['linear', 'rbf', 'sigmoid'],
        # }
        #
        # grid_search = model_selection.GridSearchCV(pipeline,
        #                                            param_grid=h_params,
        #                                            scoring='accuracy',
        #                                            cv=5,
        #                                            n_jobs=-1)
        # # trainx = preprocessing.scale(train_x)
        # grid_search.fit(train_x, train_y)
        # best_param = grid_search.best_params_
        #
        # # h_params = {
        # #     'clf__C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
        # #     'clf__gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
        # #     'clf__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        # # }
        # #
        # # svc = svmc.SUSvm()
        # # # xtr, ytr, xte, yte = train_x, train_y, test_x, test_y
        # # # train_x = preprocessing.scale(train_x)
        # # if idx == 0:
        # #     best_param = svc.ret_best_parameter(dX=train_x, dy=train_y, param=h_params,  nfolds=5)
        # # else:
        # #     best_param = {'C': 1.0, 'gamma': 1.1, 'kernel': 'linear'}
        # # print(best_param)
        # best_param = {'C': 1.0, 'gamma': 1.1, 'kernel': 'linear'}
        # # print(best_param.get('kernel'), best_param.get('C'), best_param.get('gamma'))
        # best_svmc = svmc.SUSvm(kernel=best_param.get('kernel'),
        #                        regularization=best_param.get('C'),
        #                        gamma=best_param.get('gamma'))
        # best_svmc.add_evidence(train_x, train_y)  # train it

        h_params = {
            'C': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': [0.1, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
        }
        svc = svmc.SUSvm()
        if idx == 0:
            best_param = svc.ret_best_parameter(dX=train_x, dy=train_y, param=h_params, nfolds=2)
        else:
            best_param = {'C': 1.0, 'gamma': 1.1, 'kernel': 'linear'}
        ker, reg, gam = best_param.get('kernel'), best_param.get('C'), best_param.get('gamma')
        print(ker, reg, gam)
        fp.write("Best Parameter for SVM" + "\n")
        ke = "kernel =" + '{}'.format(ker)
        fp.write(ke + "\n")
        ke = "C =" + '{}'.format(reg)
        fp.write(ke + "\n")
        ke = "gamma =" + '{}'.format(gam)
        fp.write(ke + "\n")
        best_svmc = svmc.SUSvm(kernel=best_param.get('kernel'),
                               regularization=best_param.get('C'),
                               gamma=best_param.get('gamma'))
        best_svmc.add_evidence(train_x, train_y)  # train it
        mod = ("SupportVectorClassifier", SVC())
        models.append(mod)
        plot_learning_curve(su_model=mod[0],
                            save_fold=save_folder[idx],
                            plot_pfix=plot_prefix[idx],
                            su_estimator=mod[1],
                            xx_train=train_x,
                            yy_train=train_y)
        best_pred_y = best_svmc.model_prediction(test_x)
        tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, best_pred_y)

        # # plot the decision function for best parameters
        # plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=plt.cm.Paired)
        # h = .02  # step size in the mesh
        # # plot the decision function
        # ax = plt.gca()
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        # # create a mesh to plot in
        # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                      np.arange(y_min, y_max, h))
        # xy = np.vstack([xx.ravel(), yy.ravel()]).T
        # Z = l_svmc.ret_decision_function(xy)
        # # plot decision boundary and margins
        # ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
        #            linestyles=['--', '-', '--'])
        # # plot support vectors
        # ax.scatter(learner.model.support_vectors_[:, 0], learner.model.support_vectors_[:, 1], s=100,
        #            linewidth=1, facecolors='none', edgecolors='k')
        # plt.show()

        tr_scores = []
        te_scores = []
        c_matrixs = []
        # # we create an instance of SVM and fit out data. We do not scale our
        # # data since we want to plot the support vectors
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        l_svmc = svmc.SUSvm(kernel='linear',
                            regularization=1.0)
        l_svmc.add_evidence(train_x, train_y)  # train it
        # Predict for test data
        pred_y = learner.model_prediction(test_x)  # get the predictions
        pred_train_y = learner.model_prediction(train_x)  # get the predictions
        tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
        tr_scores.append(tr_score)
        te_scores.append(te_score)
        c_matrixs.append(c_matrix)
        rbf_svmc = svmc.SUSvm(kernel='rbf',
                              gamma=0.7,
                              regularization=1.0)
        rbf_svmc.add_evidence(train_x, train_y)  # train it
        pred_y = learner.model_prediction(test_x)  # get the predictions
        pred_train_y = learner.model_prediction(train_x)  # get the predictions
        tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
        tr_scores.append(tr_score)
        te_scores.append(te_score)
        c_matrixs.append(c_matrix)
        poly_svmc = svmc.SUSvm(kernel='poly',
                               gamma=0.7,
                               regularization=1.0)
        poly_svmc.add_evidence(train_x, train_y)  # train it
        pred_y = learner.model_prediction(test_x)  # get the predictions
        pred_train_y = learner.model_prediction(train_x)  # get the predictions
        tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
        tr_scores.append(tr_score)
        te_scores.append(te_score)
        c_matrixs.append(c_matrix)
        sig_svmc = svmc.SUSvm(kernel='sigmoid',
                              gamma=0.7,
                              regularization=1.0)
        sig_svmc.add_evidence(train_x, train_y)  # train it
        pred_y = learner.model_prediction(test_x)  # get the predictions
        pred_train_y = learner.model_prediction(train_x)  # get the predictions
        tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
        tr_scores.append(tr_score)
        te_scores.append(te_score)
        c_matrixs.append(c_matrix)
        plot_data(save_folder[idx], plot_prefix[idx], mod[0], "Kernels", kernels, tr_scores, te_scores, c_matrixs)
        #
        # # title for the plots
        # titles = ['SVC-linear kernel',
        #           'SVC-RBF kernel',
        #           'SVC-polynomial(degree 3) kernel',
        #           'SVC-sigmoid']
        #
        # for i, clf in enumerate((l_svmc, rbf_svmc, poly_svmc, sig_svmc)):
        #     # Plot the decision boundary. For that, we will assign a color to each
        #     # point in the mesh [x_min, x_max]x[y_min, y_max].
        #     plt.subplot(2, 2, i + 1)
        #     plt.subplots_adjust(wspace=0.5, hspace=0.5)
        #
        #     Z = clf.model_prediction(np.c_[xx.ravel(), yy.ravel()])
        #
        #     # Put the result into a color plot
        #     Z = Z.reshape(xx.shape)
        #     plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        #
        #     # Plot also the training points
        #     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        #     plt.xlabel('Sepal length')
        #     plt.ylabel('Sepal width')
        #     plt.xlim(xx.min(), xx.max())
        #     plt.ylim(yy.min(), yy.max())
        #     plt.xticks(())
        #     plt.yticks(())
        #     plt.title(titles[i] + ": " + plot_prefix[idx])
        #
        # plt.show()
        # train_x, train_y, test_x, test_y = xtr, ytr, xte, yte


        # 5. Neural Networks
        print("Neural Networks")
        fp.write("\n" +"\n" + "Neural Networks" + "\n")
        hid_layers = [(10, 10), (10, 20), (10, 10, 10)]
        alphas = 0.0005
        episodes = [1000, 1500, 2000]
        scores = {}
        scores_list = []
        x_val = []
        for element in hid_layers:
            for episode in episodes:
                val = (element, episode)
                x_val.append(val)
        # print(x_value)
        x_value = [i for i, j in enumerate(x_val)]
        x_name = "HiddenLayers+Episodes"
        tr_scores = []
        te_scores = []
        c_matrixs = []
        saved_n = False
        print("Neural Networks - Classification Report")
        fp.write("Neural Networks - Classification Report" + "\n")
        for i, hl in enumerate(hid_layers):
            for j, episode in enumerate(episodes):
                val = (hl, episode)
                # create a learner and train it
                learner = snn.SUNeuralNetworks(hidden_layer=hl,
                                               episode=episode)
                learner.add_evidence(train_x, train_y)  # train it
                # Predict for test data
                pred_y = learner.model_prediction(test_x)  # get the predictions
                pred_train_y = learner.model_prediction(train_x)  # get the predictions
                tr_score, te_score, c_matrix = accuracy_test(fp, train_y, pred_train_y, test_y, pred_y)
                tr_scores.append(tr_score)
                te_scores.append(te_score)
                c_matrixs.append(c_matrix)

                if not saved_n:
                    mod = ("NeuralNetworkClassifier", MLPClassifier())
                    models.append(mod)
                    plot_learning_curve(su_model=mod[0],
                                        save_fold=save_folder[idx],
                                        plot_pfix=plot_prefix[idx],
                                        su_estimator=mod[1],
                                        xx_train=train_x,
                                        yy_train=train_y)
                    saved_n = True

                scores[i, j] = te_score
                scores_list.append(te_score)
                class_report = metrics.classification_report(test_y, pred_y)
                print(class_report)

                # print("Classification report:", class_report)
                xval = "((HiddenLayer),episode) = " + str(val)
                # print(xval)
                fp.write("Classification report: ")
                fp.write("".join(xval) + "\n")
                for line in class_report:
                    fp.write("".join(line))
            fp.write("\n" + "Attributes of the Learned Classifier" + "\n")
            fp.write("current loss computed with the loss function: ")
            fp.write("".join(str(learner.model.loss_)) + "\n")
            fp.write("number of iterations the solver: ")
            fp.write("".join(str(learner.model.n_iter_)) + "\n")
            fp.write("num of layers: ")
            fp.write("".join(str(learner.model.n_layers_)) + "\n")
            fp.write("Num of o/p: ")
            fp.write("".join(str(learner.model.n_outputs_)) + "\n")

        plot_data(save_folder[idx], plot_prefix[idx], mod[0], x_name, x_value, tr_scores, te_scores, c_matrixs)

        # All models Evaluation for the given data set with default settings of Classifiers in SciKit Learn Library
        # # evaluate each model in turn
        results = []
        names = []
        scoring = 'accuracy'
        print("SciKit Learn Library - Supervised Learning algorithms comparison with Default settings")
        fp.write("SciKit Learn Library - Supervised Learning algorithms comparison with Default settings" + "\n")
        for name, model in models:
            splits = 4
            kfold = model_selection.KFold(n_splits=splits)
            cv_results = model_selection.cross_val_score(model, train_x, train_y, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
            fp.write(msg + "\n")
        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle(plot_prefix[idx] + ' - Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_ylabel("score")
        xnames = ['DTC', 'ADB', 'KNN', 'SVM', 'NN']
        ax.set_xticklabels(xnames)
        if win:
            plt.savefig("results\\" + plot_prefix[idx] + "-algorithm_compared.png")
        else:
            plt.savefig("results/" + plot_prefix[idx] + "-algorithm_compared.png")
        plt.close()
        # plt.show()
        fp.write("\n" + "\n" + "\n")

    # After all iterations for both Datasets
    plt.close('all')
    fp.write("Done")
    fp.close()
    print("Done")

