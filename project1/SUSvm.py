#!/usr/bin/env python3
import numpy as np

"""
Import the DecisionTreeClassifier model.
"""
from sklearn import svm
from sklearn import model_selection
# scaling the data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

class SUSvm(object):
    """
    This is a Boosting Learner - Using Correlation as per JR Quinlan method.

    :param leaf_size:  is a hyperparameter that defines the maximum number of samples to be aggregated at a leaf.
    :type leaf_size: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, kernel='linear', gamma='auto', regularization=1.0, degree=3):
        """
        :param leaf_size:
        :param verbose:
        """
        self.learner_model = None
        self.model = None
        self.kernel = kernel
        self.gamma = gamma # Kernel coefficient
        self.regularization = regularization  # C parameter, always +ve
        self.degree = degree
        self.svc = None
        self.model = svm.SVC(kernel=self.kernel,
                             C=self.regularization,
                             gamma=self.gamma,
                             degree=self.degree)

        # self.pipeline = Pipeline([
        #     ('vect', CountVectorizer()),
        #     ('tfidf', TfidfTransformer()),
        #     ('clf', svm.SVC(kernel=self.kernel,
        #                      C=self.regularization,
        #                      gamma=self.gamma,
        #                      degree=self.degree))
        # ])

    def fit(self, train_features, train_targets):
        return self.model.fit(train_features, train_targets)

    def ret_best_parameter(self, dX, dy, param, nfolds=5):
        self.svc = svm.SVC()
        # pipe_svm = Pipeline([('vect', CountVectorizer()),
        #                      ('tfidf', TfidfTransformer()),
        #                      ('clf', svm.SVC())])
        #
        # grid_search = model_selection.GridSearchCV(pipe_svm,
        #                                            param_grid=param,
        #                                            scoring='accuracy',
        #                                            cv=nfolds,
        #                                            n_jobs=-1)
        # # print("pipesvm", pipe_svm.get_params().keys())
        # trainx = preprocessing.scale(dX)
        # grid_search.fit(trainx, dy)
        # parameter = grid_search.best_params_
        grid_search = model_selection.GridSearchCV(self.svc,
                                                   param_grid=param,
                                                   scoring='accuracy',
                                                   cv=nfolds)
        grid_search.fit(dX, dy)
        parameter = grid_search.best_params_
        return parameter

    def build_tree(self, data):
        """
        Train the model
        """
        train_features = data[:, 0:-1]
        train_targets = data[:, -1]
        train_features = preprocessing.scale(train_features)
        self.learner_model = self.model.fit(train_features, train_targets)

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # slap on 1s column so linear regression finds a constant term
        new_data = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data[:, 0 : data_x.shape[1]] = data_x
        # build and save the model
        new_data[:, -1] = data_y
        self.build_tree(new_data)

    def model_prediction(self, test_features):
        test_features = preprocessing.scale(test_features)
        prediction = self.learner_model.predict(test_features)
        return prediction

    def ret_accuracy_score(self, data_x, data_y):
        return self.learner_model.score(data_x, data_y)

    def ret_decision_function(self, data):
        return self.model.decision_function(data)


if __name__ == "__main__":
    print(" ML SVM")

