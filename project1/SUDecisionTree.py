#!/usr/bin/env python3
import numpy as np

"""
Import the DecisionTreeClassifier model.
"""

#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


class SUDecisionTree(object):
    """
    This is a Decision Tree Learner - Using Correlation as per JR Quinlan method.

    :param leaf_size:  is a hyperparameter that defines the maximum number of samples to be aggregated at a leaf.
    :type leaf_size: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, leaf_size=1, features=5, modeltype=0, c_criteria=1):
        """
        :param leaf_size:
        :param verbose:
        """
        self.learner_model = None
        self.leaf_size = leaf_size
        self.features = features
        self.model_type = modeltype
        self.model_classifier = 0
        self.model_regressor = 1
        if c_criteria == 0:
            self.ccriteria = "gini"
        else:
            self.ccriteria = "entropy"
        if self.model_type == self.model_classifier:
            self.model = DecisionTreeClassifier(criterion=self.ccriteria,
                                                min_samples_leaf=self.leaf_size,
                                                max_features=self.features)
        else:
            self.model = DecisionTreeRegressor(criterion='mse',
                                               min_samples_leaf=self.leaf_size,
                                               max_features=self.features)

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
    print("ML Decision Tree")
