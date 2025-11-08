##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 7                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 12-01-2023
# Updated by Ameni Ben Salah on 19-03-2025

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import copy
import os
import time


class ClassificationAlgorithms:

    # Apply a decision tree approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def decision_tree(
        self,
        train_X,
        train_y,
        test_X,
        min_samples_leaf=50,
        criterion="gini",
        print_model_details=False,
        export_tree_path="../../reports/Models",
        export_tree_name="tree.dot",
        gridsearch=True,
    ):
        # Create the model
        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "criterion": ["gini", "entropy"],
                }
            ]
            dtree = GridSearchCV(
                DecisionTreeClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            dtree = DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf, criterion=criterion
            )

        # Fit the model

        dtree.fit(train_X, train_y.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        start = time.time()
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        end = time.time()
        pre_time = end - start
        frame_prob_training_y = pd.DataFrame(
            pred_prob_training_y, columns=dtree.classes_
        )
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(dtree.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
            print("Feature importance decision tree:")
            for i in range(0, len(dtree.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(dtree.feature_importances_[ordered_indices[i]])

            # ✅ Ensure the directory exists before exporting the tree
            os.makedirs(export_tree_path, exist_ok=True)

            tree.export_graphviz(
                dtree,
                out_file=os.path.join(
                    export_tree_path, export_tree_name
                ),  # More robust path handling
                feature_names=train_X.columns,
                class_names=[
                    str(cls) for cls in dtree.classes_
                ],  # Ensure class names are strings
            )

        return (
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
            pre_time,
        )

    # Apply a naive bayes approach for classification upon the training data
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def naive_bayes(self, train_X, train_y, test_X):
        # Create the model
        nb = GaussianNB()

        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        start = time.time()
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(test_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)
        end = time.time()
        pred_time = end - start
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)

        return (
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
            pred_time,
        )

    # Apply a random forest approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def random_forest(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=10,
        min_samples_leaf=5,
        criterion="gini",
        print_model_details=False,
        gridsearch=False,
    ):

        if gridsearch:
            tuned_parameters = [
                {
                    "min_samples_leaf": [2, 10, 50, 100, 200],
                    "n_estimators": [10, 50, 100],
                    "criterion": ["gini", "entropy"],
                }
            ]
            rf = GridSearchCV(
                RandomForestClassifier(), tuned_parameters, cv=5, scoring="accuracy"
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
            )

        # Fit the model

        rf.fit(train_X, train_y.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        start = time.time()

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)

        end = time.time()
        pred_time = end - start
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [
                i[0]
                for i in sorted(
                    enumerate(rf.feature_importances_), key=lambda x: x[1], reverse=True
                )
            ]
            print("Feature importance random forest:")
            for i in range(0, len(rf.feature_importances_)):
                print(
                    train_X.columns[ordered_indices[i]],
                )
                print(
                    " & ",
                )
                print(rf.feature_importances_[ordered_indices[i]])

        return (
            pred_training_y,
            pred_test_y,
            frame_prob_training_y,
            frame_prob_test_y,
            pred_time,
        )

    # Extreme Gradient Boosting (XGBoost)
    def xgboost_classifier(
        self,
        train_X,
        train_y,
        test_X,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        gridsearch=True,
        print_model_details=True,
    ):
        if gridsearch:
            tuned_parameters = [
                {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                }
            ]
            xgb = GridSearchCV(
                XGBClassifier(eval_metric="logloss"),
                tuned_parameters,
                cv=5,
                scoring="accuracy",
            )
        else:
            xgb = XGBClassifier(
                learning_rate=learning_rate,  # remplace eta
                max_depth=max_depth,
                n_estimators=n_estimators,
                eval_metric="logloss",
            )

        xgb.fit(train_X, train_y.ravel())

        if gridsearch and print_model_details:
            print(xgb.best_params_)

        if gridsearch:
            xgb = xgb.best_estimator_

        start = time.time()

        pred_training_y = xgb.predict(train_X)
        pred_test_y = xgb.predict(test_X)

        end = time.time()
        pred_time = end - start

        return (
            pred_training_y,
            pred_test_y,
            pred_time,
        )
