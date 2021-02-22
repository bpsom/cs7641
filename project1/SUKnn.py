#!/usr/bin/env python3
import numpy as np

"""
Import the DecisionTreeClassifier model.
"""
from sklearn.neighbors import KNeighborsClassifier


class SUKnn(object):
    """
    This is a Boosting Learner - Using Correlation as per JR Quinlan method.

    :param leaf_size:  is a hyperparameter that defines the maximum number of samples to be aggregated at a leaf.
    :type leaf_size: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, k=1):
        """
        :param leaf_size:
        :param verbose:
        """
        self.learner_model = None
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

    def fit(self, train_features, train_targets):
        return self.model.fit(train_features, train_targets)

    def build_tree(self, data):
        """
        Train the model
        """
        train_features = data[:, 0:-1]
        train_targets = data[:, -1]
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
        prediction = self.learner_model.predict(test_features)
        return prediction

    def ret_accuracy_score(self, data_x, data_y):
        return self.learner_model.score(data_x, data_y)


if __name__ == "__main__":
    print(" ML K-Nearest Neighbors")
