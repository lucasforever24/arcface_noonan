'''
This file contain all the models used in training using
dataset [drebin], we evaluate the metrices using sklearn metrices

Training with different models (SVM (c_values = [1, 10, 100, 1000]),
Recursive Feature Elimination, naive Bayes,
Random Forest (n_jobs = [None, 10] & n_estimators = [10, 1000]),
Extra Trees (n_estimators = [10, 100, 500, 1000] & nJobs = [100, 500, 1000]),
grid search based on SVC model (kernel = [linear, rbf] & c = [1, 10])
grid search based on RF model (n_estimators = [200, 500] &
  max_features: [auto, sqrt, log2] &
  max_depth: [4, 5, 6, 7, 8] &
  criterion: [gini, entropy])
'''

import time
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, classification_report,
                             average_precision_score)
import dask_ml.model_selection as dcv
# from dask.diagnostics import ProgressBar
# from sklearn.model_selection import ParameterGrid


# Used to print the performance metrices for whatever model invokes this method
def print_metrices_out(y_predicted, y_test, y_prob, label_bin, avg='micro'):
    print("Accuracy is %f (in percentage)" %
          (accuracy_score(y_test, y_predicted) * 100))
    print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_predicted)))
    print("Recall score is %f." % recall_score(y_test, y_predicted, average=avg))
    print("Precision score is %f." %
          precision_score(y_test, y_predicted, average=avg))
    print("F1 score is %f." % f1_score(y_test, y_predicted, average=avg))
    test_auc = metrics.roc_auc_score(y_test, y_prob, average=avg)
    print("AUC score is %f." % test_auc)
    test_ap = average_precision_score(y_test, y_prob, average=avg)
    print('Average precison score is %f' % test_ap)
    print("classification Report: \n" +
          str(classification_report(y_test, y_predicted)))
    print("-----------------------------------\n")


# Used to print the performance metrices for whatever model invokes this method
def print_metrices_multiclass(y_predicted, y_test, y_prob, label_bin, avg='micro'):
    print("Accuracy is %f (in percentage)" %
          (accuracy_score(y_test, y_predicted) * 100))
    print("Confusion Matrix: \n" + str(confusion_matrix(y_test, y_predicted)))
    print("Recall score is %f." % recall_score(y_test, y_predicted, average=avg))
    print("Precision score is %f." %
          precision_score(y_test, y_predicted, average=avg))
    print("F1 score is %f." % f1_score(y_test, y_predicted, average=avg))
    test_auc = metrics.roc_auc_score(label_bin, y_prob, average=avg)
    print("AUC score is %f." % test_auc)
    test_ap = average_precision_score(label_bin, y_prob, average=avg)
    print('Average precison score is %f' % test_ap)
    print("classification Report: \n" +
          str(classification_report(y_test, y_predicted)))
    print("-----------------------------------\n")

def get_auc(y_test, y_prob):
    test_auc = metrics.roc_auc_score(y_test, y_prob[:, 1])
    return test_auc


# This section contains the fitting of data in the model
# and the prediction of the test data passed to the parameter
# Note that you can add n_jobs=-1 to allow the model to use
# all the preprocessors offered by your pc,
# for example with grid search CV which is known for its
# expensive computations it took 6 mins instead of 15mins
# given same dataset, same PC
def train_svm(x_train, y_train, x_test, y_test):
    print("\n-------------SVM Model-------------")
    model = SVC(gamma='scale', probability=True, kernel='poly')
    # SVM Fit
    model.fit(x_train, y_train)
    # SVM Predict
    y_predicted = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    # print("SVM Evaluation parameters:")
    # print_metrices_out(y_predicted, y_test, y_prob)
    return y_predicted, y_prob


def train_svm_tuning_c_val(x_train, y_train, x_test, y_test, c_value):
    print("-------------SVM, C value Model-------------")
    print("C value: " + str(c_value))
    class_weight = dict()
    class_weight[1] = 1
    class_weight[0] = 1
    model = SVC(gamma='scale', C=c_value, probability=True, class_weight=class_weight)
    # SVM Fit
    model.fit(x_train, y_train)
    # SVM Predict
    y_predicted = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    return y_predicted, y_prob


def train_logistic_regression(x_train, y_train, x_test, y_test):
    print("-------------RFE Model-------------")
    class_weight = dict()
    class_weight[1] = 1
    class_weight[0] = 1
    model = LogisticRegression(solver='sag', class_weight=class_weight, max_iter=150)
    # RFE Fit
    model.fit(x_train, y_train)
    coef = model.coef_
    # RFE Predict
    y_predicted = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    return y_predicted, y_prob

def train_recursive_feature_elimination(x_train, y_train, x_test, y_test, feature_num=10):
    print("-------------RFE Model-------------")
    class_weight = dict()
    class_weight[1] = 1
    class_weight[0] = 1
    model = LogisticRegression(solver='sag', class_weight=class_weight)
    # model = RandomForestClassifier(n_estimators=100)
    # model = SVC(gamma='scale', probability=True, kernel='poly')
    rfe = RFE(model, feature_num)
    # RFE Fit
    rfe.fit(x_train, y_train)
    # RFE Predict
    y_predicted = rfe.predict(x_test)
    y_prob = rfe.predict_proba(x_test)
    print(rfe.support_)
    return y_predicted, y_prob

def train_extra_trees(x_train, y_train, x_test, y_test):
    print("-------------Extra Trees Model-------------")
    extra_trees = ExtraTreesClassifier(n_estimators=100)
    # ET Fit
    extra_trees.fit(x_train, y_train)
    # ET Predict
    y_predicted = extra_trees.predict(x_test)
    y_prob = extra_trees.predict_proba(x_test)
    # ET Matrices
    # print("ET Evaluation parameters:")
    # print_metrices_out(y_predicted, y_test, y_prob)
    return y_predicted, y_prob


def train_extra_trees_n_estimators(x_train, y_train, x_test, y_test,
                                   number_estimators):
    print("-------------Extra Trees Model-------------")
    print("Number of estimators: " + str(number_estimators))
    class_weight = dict()
    class_weight[1] = 1
    class_weight[0] = 1
    extra_trees = ExtraTreesClassifier(n_estimators=number_estimators)
    # ET Fit
    extra_trees.fit(x_train, y_train)
    # ET Predict
    y_predicted = extra_trees.predict(x_test)
    y_prob = extra_trees.predict_proba(x_test)
    # ET Matrices
    # print("ET Evaluation parameters:")
    # print_metrices_out(y_predicted, y_test)
    return y_predicted, y_prob


def train_extra_trees_n_jobs(x_train, y_train, x_test, y_test, n_jobs):
    print("-------------Extra Trees Model-------------")
    print("Number of jobs: " + str(n_jobs))
    extra_trees = ExtraTreesClassifier(n_jobs=n_jobs)
    # ET Fit
    extra_trees.fit(x_train, y_train)
    # ET Predict
    y_predicted = extra_trees.predict(x_test)
    # ET Matrices
    print("ET Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)


def train_rf(x_train, y_train, x_test, y_test):
    print("-------------RF Model-------------")
    class_weight = dict()
    class_weight[1] = 1
    class_weight[0] = 1
    model = RandomForestClassifier(n_estimators=100)
    # RF Fit
    model.fit(x_train, y_train)
    # RF Predict
    y_predicted = model.predict(x_test)
    # RF Matrices
    print("RF Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)


def train_rf_number_jobs(x_train, y_train, x_test, y_test, number_of_jobs):
    print("-------------RF Model with nJobs-------------")
    print("N_Jobs: " + str(number_of_jobs))
    model = RandomForestClassifier(n_estimators=100, n_jobs=number_of_jobs)
    # RF Fit
    model.fit(x_train, y_train)
    # RF Predict
    y_predicted = model.predict(x_test)
    # RF Matrices
    print("RF Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)


def train_rf_number_jobs_estimators(x_train, y_train, x_test, y_test,
                                    numbers_estimators=100, number_of_jobs=10):
    print("-------------RF Model with nJobs-------------")
    print("N_Jobs: " + str(number_of_jobs))
    print("N_Estimators: " + str(numbers_estimators))
    class_weight = dict()
    class_weight[1] = 100
    class_weight[0] = 1
    model = RandomForestClassifier(
        n_estimators=numbers_estimators, n_jobs=number_of_jobs)
    # RF Fit
    t_start = time.time()
    model.fit(x_train, y_train)
    t_finish = time.time()
    print(round((t_finish - t_start), 2), "Time to finish training RF with 10 nJobs\n")
    # RF Predict
    y_predicted = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    # RF Matrices
    # print("RF Evaluation parameters:")
    # print_metrices_out(y_predicted, y_test, y_prob)
    return y_predicted, y_prob


def train_naive_bayes(x_train, y_train, x_test, y_test):
    print("-------------NB Model-------------")
    bern_naive_bayes = BernoulliNB()
    # NB Fit
    bern_naive_bayes.fit(x_train, y_train)
    # NB Predict
    y_predicted = bern_naive_bayes.predict(x_test)
    # NB Matrices
    print("NB Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)


def train_grid_search_using_svm(x_train, y_train, x_test, y_test):
    print("-------------GS, SVC Model-------------")
    svc = SVC(gamma="scale")
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    # to print out the number combinatoric product of the different parameters
    # so if you have cv of 5 - as it is the case here -
    # within each grid you will have 5cross validations thus
    # thus we have 20 number of fits
    # to use it uncomment this import sklearn.model_selection
    # pg = ParameterGrid(parameters)
    # print(len(pg))
    clf = GridSearchCV(svc, parameters, cv=5,
                       refit=True, n_jobs=-1, verbose=1)
    # GS, SVC Fit
    clf.fit(x_train, y_train)
    # GS, SVC Predict
    y_predicted = clf.predict(x_test)
    # GS, SVC Matrices
    print("GS based on SVC model Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)


def train_grid_search_using_rf(x_train, y_train, x_test, y_test):
    print("-------------GS, RF Model-------------")
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    clf = dcv.GridSearchCV(estimator=RandomForestClassifier(verbose=3), cv=5,
                           refit=True,
                           param_grid=param_grid,
                           error_score=0)
    # GS, RF Fit
    # use of progress bar to print out "progress"
    # [########################################] | 100% Completed |  4min 11.2s
    # to use it uncomment this import dask.diagnostics
    # with ProgressBar(dt=60.0):
    clf.fit(x_train, y_train)
    # GS, RF Predict
    y_predicted = clf.predict(x_test)
    # GS, RF Matrices
    print("GS based on RF model Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)


def train_gbdt(x_train, y_train, x_test, y_test):
    print("-------------GBDT Model-------------")
    model = GradientBoostingClassifier()
    # RF Fit
    model.fit(x_train, y_train)
    # RF Predict
    y_predicted = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    # RF Matrices
    print("RF Evaluation parameters:")
    return y_predicted, y_prob


def train_grid_search_using_gdbt(x_train, y_train, x_test, y_test):
    print("-------------GS, GBDT Model-------------")
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.5]
    }
    clf = dcv.GridSearchCV(estimator=GradientBoostingClassifier(verbose=3), cv=3,
                           refit=True,
                           param_grid=param_grid,
                           error_score=0)
    # GS, RF Fit
    # use of progress bar to print out "progress"
    # [########################################] | 100% Completed |  4min 11.2s
    # to use it uncomment this import dask.diagnostics
    # with ProgressBar(dt=60.0):
    clf.fit(x_train, y_train)
    print(clf.get_params())
    # GS, RF Predict
    y_predicted = clf.predict(x_test)
    # GS, RF Matrices
    print("GS based on RF model Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)


def train_knn(x_train, y_train, x_test, y_test, n_neighbors=5):
    print("-------------knn Model-------------")
    class_weight = dict()
    class_weight[1] = 8
    class_weight[0] = 1
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # RF Fit
    model.fit(x_train, y_train)
    # RF Predict
    y_predicted = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    # RF Matrices
    # print("RF Evaluation parameters:")
    return y_predicted, y_prob


def train_grid_search_using_knn(x_train, y_train, x_test, y_test):
    print("-------------GS, GBDT Model-------------")
    param_grid = {
        'n_neighbors': [2, 5, 10, 20],
        'p': [1, 2],
        'leaf_size': [10, 20, 25, 30, 50]
    }
    clf = dcv.GridSearchCV(estimator=KNeighborsClassifier(verbose=3), cv=3,
                           refit=True,
                           param_grid=param_grid,
                           error_score=0)
    # GS, RF Fit
    # use of progress bar to print out "progress"
    # [########################################] | 100% Completed |  4min 11.2s
    # to use it uncomment this import dask.diagnostics
    # with ProgressBar(dt=60.0):
    clf.fit(x_train, y_train)
    print(clf.get_params())
    # GS, RF Predict
    y_predicted = clf.predict(x_test)
    # GS, RF Matrices
    print("GS based on RF model Evaluation parameters:")
    print_metrices_out(y_predicted, y_test)