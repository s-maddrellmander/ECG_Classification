import logging

import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import Timer


def describe_data(X, y):
    logging.info("\n" + "="*50 + "\n\tDataset Shapes\n" + "="*50)
    logging.info(f"X shape: {X.shape}")
    if type(y) is list:
        logging.info(f"y shape: ({len(y)}, 1)")
    else:
        # TODO: Assuming array
        logging.info(f"y shape: {y.shape}")
    

def simple_decision_tree_calssifier(X_train, y_train, X_test, y_test):
    # training a DescisionTreeClassifier
    y_train = y_train.tolist()
    num_samples, shape_i, shape_j = X_train.shape
    X_train = X_train.reshape((num_samples, shape_i*shape_j))

    y_test = y_test.tolist()
    num_samples, shape_i, shape_j = X_test.shape
    X_test = X_test.reshape((num_samples, shape_i*shape_j))
    
    describe_data(X_train, y_train)
    
    with Timer("DecisionTreeClassifier"):
        clf = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
        # creating a confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_pred, y_test)
        # roc = roc_auc_score(y_pred, y_test)
        logging.info(f"Percentage correct:  {100*np.sum(y_pred == y_test)/len(y_test):.4f}")
        # logging.info(f"ROC Score: {roc:.5f}")
    
    # with Timer("SGD Classifier"):
    #     sgd_clf = SGDClassifier(random_state=42, max_iter=10, tol=1e-3, verbose=0)
    #     sgd_clf.fit(X_train, y_train)
        
    #     y_pred = sgd_clf.predict(X_test)
    #     # roc = roc_auc_score(y_pred, y_test)
    #     logging.info(f"Percentage correct:  {100*np.sum(y_pred == y_test)/len(y_test):.4f}")
    #     # logging.info(f"ROC Score: {roc:.5f}")