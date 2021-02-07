import sys
sys.path.append('../')

import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from utils import label_binarize


from ml_noonan.train_model import train_logistic_regression
# from mlearning_classification.utils import get_correct_number
from ml_noonan.train_model import get_auc
from ml_noonan.train_model import train_recursive_feature_elimination
from ml_noonan.train_model import train_extra_trees
from ml_noonan.train_model import train_rf_number_jobs_estimators
from ml_noonan.train_model import train_svm
from ml_noonan.train_model import train_gbdt
from ml_noonan.train_model import train_knn
from ml_noonan.train_model import print_metrices_out, print_metrices_multiclass


def experiment(data_path):
    data = np.load(data_path)
    x = data[:, 1:]
    y = data[:, 0]

    experiment_time = 1
    for i in range(experiment_time):
        y_total_predicted = []
        y_total_label = []
        y_total_prob = []
        correct_list = []

        print("======================")
        print("Experiment:", i)
        print("======================")
        kf = KFold(n_splits=10, shuffle=True)
        k = 1

        for train_index, test_index in kf.split(x):
            print('10-fold Train, number:', k)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print('\nTraining data shape (x, y): ' +
                  str(x_train.shape), str(y_train.shape))

            print('\nTesting data shape (x, y): ' +
                  str(x_test.shape), str(y_test.shape))
            print("Number of noonan for test:", sum(y_test))

            y_predicted, y_prob = train_svm(x_train, y_train, x_test, y_test)

            if k == 1:
                y_total_predicted = y_predicted
                y_total_label = y_test  ###########
                y_total_prob = y_prob
            else:
                y_total_predicted = np.concatenate((y_predicted, y_total_predicted))
                y_total_label = np.concatenate((y_test, y_total_label))  ##########
                y_total_prob = np.concatenate((y_prob, y_total_prob))

            k += 1

            # y_total_predicted = np.array(y_total_predicted)
            # y_total_label = np.array(y_total_label)
            # y_total_prob = np.array(y_total_prob)

        label_bin = label_binarize(y_total_label, classes=[0, 1])
        scores = y_total_prob.ravel()
        labels = label_bin.ravel()
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)

        print_metrices_multiclass(y_total_predicted, y_total_label, y_total_prob, label_bin, avg='macro')

        np.save("scores.npy", scores)
        np.save("labels.npy", labels)
        print("result saved!")


if __name__ == "__main__":
    data_path = 'otnn.npy'
    experiment(data_path)


