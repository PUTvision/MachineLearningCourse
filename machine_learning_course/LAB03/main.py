from sklearn import datasets, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd
import pickle
import itertools
import numpy as np


def check_different_classificator(clf, clf_name, x_train, x_test, y_train, y_test, gs, grid, figure, plot=False):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    # score = metrics.accuracy_score()
    if plot:
        try:
            ax = plt.subplot(gs[grid[0], grid[1]])
            figure = plot_decision_regions(x_train, y_train.values, clf=clf, legend=2)
            plt.title(clf_name)
        except ValueError:
            ax = plt.subplot(gs[grid[0], grid[1]])
            figure = plot_decision_regions(x_train.values, y_train.values, clf=clf, legend=2)
            plt.title(clf_name)

    return score


def todo1():
    iris = datasets.load_iris(as_frame=True)
    print(iris.frame.describe())

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42,
                                                        stratify=iris.target)

    min_max_scaler = MinMaxScaler()
    # Dobieramy i dopasowujemy te dane tylko na zbiorze treningowym. Transform należy używać już na obywdówch zbiorach
    min_max_scaler.fit(x_train)
    x_train_min_max_scaled = min_max_scaler.transform(x_train)

    standard_scaler = StandardScaler()
    standard_scaler.fit(x_train)
    x_train_standard_scaled = standard_scaler.transform(x_train)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(x_train.loc[:, "sepal length (cm)"], x_train.loc[:, "sepal width (cm)"], color='r')
    axs[0, 0].set_title('Dane wejsciowe')
    axs[0, 0].set_xlabel('Sepal length (m)')
    axs[0, 0].set_ylabel('Sepal width (m)')
    axs[0, 0].axvline(x=0)
    axs[0, 0].axhline(y=0)

    axs[0, 1].scatter(x_train_standard_scaled[:, 0], x_train_standard_scaled[:, 1], color='g')
    axs[0, 1].set_title('Dane po standaryzacji')
    axs[0, 1].set_xlabel('Sepal length')
    axs[0, 1].set_ylabel('Sepal width')
    axs[0, 1].axvline(x=0)
    axs[0, 1].axhline(y=0)

    axs[1, 0].scatter(x_train_min_max_scaled[:, 0], x_train_min_max_scaled[:, 1], color='b')
    axs[1, 0].set_title('Dane po min-max scaling')
    axs[1, 0].set_xlabel('Sepal length')
    axs[1, 0].set_ylabel('Sepal width')
    axs[1, 0].axvline(x=0)
    axs[1, 0].axhline(y=0)

    fig.suptitle('Iris features')
    plt.show()

    clf = svm.SVC()
    clf.fit(x_train_standard_scaled[:, 0:2], y_train)
    plot_decision_regions(x_train_standard_scaled[:, 0:2], y_train.values, clf=clf, legend=2)

    plt.xlabel('sepal length normalized')
    plt.ylabel('sepal length normalized')
    plt.title('SVM on Iris')
    plt.show()


def todo2():
    iris = datasets.load_iris(as_frame=True)
    print(iris.frame.describe())

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42,
                                                        stratify=iris.target)

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(x_train)

    standard_scaler = StandardScaler()
    standard_scaler.fit(x_train)

    x_train_standard_scaled = standard_scaler.transform(x_train)
    x_test_standard_scaled = standard_scaler.transform(x_test)

    x_train_min_max_scaled = min_max_scaler.transform(x_train)
    x_test_min_max_scaled = min_max_scaler.transform(x_test)

    classificators = [svm.SVC(), LinearRegression(), RandomForestClassifier(), DecisionTreeClassifier()]
    classificators_names = ["svc", "linear_regression", "random_forest", "decision_tree"]

    score_dict = {}

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))
    for classifier, clf_name, grd in zip(classificators, classificators_names, itertools.product([0, 1], repeat=2)):
        score_dict[clf_name] = check_different_classificator(classifier, clf_name,
                                                             x_train.loc[:, "sepal length (cm)":"sepal width (cm)"],
                                                             x_test.loc[:, "sepal length (cm)":"sepal width (cm)"],
                                                             y_train, y_test, gs, grd, fig, plot=True)
    print(f"Wyniki klasyfikatorow wytrenowanych na nadanych nieprzeskalowanych: {score_dict}")
    plt.suptitle("Pola decyzyjne dla klasyfikatorów wytrenowanych na danych nieprzeskalowanych")

    fig1 = plt.figure(figsize=(10, 8))
    for classifier, clf_name, grd in zip(classificators, classificators_names, itertools.product([0, 1], repeat=2)):
        score_dict[clf_name] = check_different_classificator(classifier, clf_name,
                                                             x_train_min_max_scaled[:, 0:2],
                                                             x_test_min_max_scaled[:, 0:2],
                                                             y_train, y_test, gs, grd, fig1, plot=True)
    print(f"Wyniki klasyfikatorow wytrenowanych nadanych po min-max scalingu: {score_dict}")
    plt.suptitle("Pola decyzyjne dla klasyfikatorów wytrenowanych na danych po min-max scalingu")

    fig2 = plt.figure(figsize=(10, 8))
    for classifier, clf_name, grd in zip(classificators, classificators_names, itertools.product([0, 1], repeat=2)):
        score_dict[clf_name] = check_different_classificator(classifier, clf_name,
                                                             x_train_standard_scaled[:, 0:2],
                                                             x_test_standard_scaled[:, 0:2],
                                                             y_train, y_test, gs, grd, fig2, plot=True)
    print(f"Wyniki klasyfikatorow wytrenowanych nadanych po standard scalingu: {score_dict}")
    plt.suptitle("Pola decyzyjne dla klasyfikatorów wytrenowanych na danych po standard scalingu")
    plt.show()


def todo3():
    iris = datasets.load_iris(as_frame=True)
    print(iris.frame.describe())

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42,
                                                        stratify=iris.target)

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(x_train)

    standard_scaler = StandardScaler()
    standard_scaler.fit(x_train)

    x_train_standard_scaled = standard_scaler.transform(x_train)
    x_test_standard_scaled = standard_scaler.transform(x_test)

    x_train_min_max_scaled = min_max_scaler.transform(x_train)
    x_test_min_max_scaled = min_max_scaler.transform(x_test)

    classificators = [svm.SVC(), LinearRegression(), RandomForestClassifier(), DecisionTreeClassifier()]
    classificators_names = ["svc", "linear_regression", "random_forest", "decision_tree"]

    score_dict = {}
    gs = gridspec.GridSpec(2, 2)
    fig1 = plt.figure(figsize=(10, 8))

    for classifier, clf_name, grd in zip(classificators, classificators_names, itertools.product([0, 1], repeat=2)):
        score_dict[clf_name] = check_different_classificator(classifier, clf_name,
                                                             x_train_min_max_scaled[:, 0:2],
                                                             x_test_min_max_scaled[:, 0:2],
                                                             y_train, y_test, gs, grd, fig1, plot=False)
    print(f"Wyniki klasyfikatorow wytrenowanych na 2 cechach: {score_dict}")

    for classifier, clf_name, grd in zip(classificators, classificators_names, itertools.product([0, 1], repeat=2)):
        score_dict[clf_name] = check_different_classificator(classifier, clf_name,
                                                             x_train_min_max_scaled[:, 0:4],
                                                             x_test_min_max_scaled[:, 0:4],
                                                             y_train, y_test, gs, grd, fig1, plot=False)
    print(f"Wyniki klasyfikatorow wytrenowanych na 4 cechach: {score_dict}")


def todo4():
    iris = datasets.load_iris(as_frame=True)
    print(iris.frame.describe())

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42,
                                                        stratify=iris.target)

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(x_train)
    x_train_min_max_scaled = min_max_scaler.transform(x_train)
    x_test_min_max_scaled = min_max_scaler.transform(x_test)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                        'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(svm.SVC(), param_grid=tuned_parameters)
    clf.fit(x_train_min_max_scaled, y_train)
    print(f"SVC best params: {clf.best_params_}")

    tuned_parameters = [{'criterion': ['gini', 'entropy'], 'splitter': ["best", "random"]}]
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid=tuned_parameters)
    clf.fit(x_train_min_max_scaled, y_train)

    print(f"DecisionTree best params: {clf.best_params_}")
    print(f"DecisionTree score: {clf.score(x_test_min_max_scaled, y_test)}")

    filename = 'svc_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    clf_loaded = pickle.load(open(filename, 'rb'))
    print(f"DecisionTree score: {clf_loaded.score(x_test_min_max_scaled, y_test)}")


todo1()
todo2()
todo3()
todo4()
