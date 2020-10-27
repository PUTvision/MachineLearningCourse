from sklearn import datasets, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd
import itertools
import numpy as np


def check_different_classificator(clf, clf_name, x_train, x_test, y_train, y_test, gs, grid, figure):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
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
    data = datasets.load_iris(as_frame=True)
    iris_dataset = pd.DataFrame(data['data'], columns=data["feature_names"])
    iris_dataset["Species"] = data['target']
    iris_dataset["Species"] = iris_dataset["Species"].apply(lambda x: data["target_names"][x])
    print(iris_dataset.head())

    x_train, x_test, y_train, y_test = train_test_split(iris_dataset.loc[:, "sepal length (cm)":"petal width (cm)"],
                                                        data['target'], test_size=0.2, random_state=42,
                                                        stratify=data['target'])

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(x_train)

    standard_scaler = StandardScaler()
    standard_scaler.fit(x_train)

    x_train_standard_scaled = standard_scaler.transform(x_train)

    x_train_min_max_scaled = min_max_scaler.transform(x_train)

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
    data = datasets.load_iris(as_frame=True)
    iris_dataset = pd.DataFrame(data['data'], columns=data["feature_names"])
    iris_dataset["Species"] = data['target']
    iris_dataset["Species"] = iris_dataset["Species"].apply(lambda x: data["target_names"][x])
    print(iris_dataset.head())

    x_train, x_test, y_train, y_test = train_test_split(iris_dataset.loc[:, "sepal length (cm)":"petal width (cm)"],
                                                        data['target'], test_size=0.2, random_state=42,
                                                        stratify=data['target'])

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
                                                             y_train, y_test, gs, grd, fig)
    print(f"Wyniki klasyfikatorow wytrenowanych na nadanych nieprzeskalowanych: {score_dict}")
    plt.suptitle("Pola decyzyjne dla klasyfikatorów wytrenowanych na danych nieprzeskalowanych")

    fig1 = plt.figure(figsize=(10, 8))
    for classifier, clf_name, grd in zip(classificators, classificators_names, itertools.product([0, 1], repeat=2)):
        score_dict[clf_name] = check_different_classificator(classifier, clf_name,
                                                             x_train_min_max_scaled[:, 0:2],
                                                             x_test_min_max_scaled[:, 0:2],
                                                             y_train, y_test, gs, grd, fig1)
    print(f"Wyniki klasyfikatorow wytrenowanych nadanych po min-max scalingu: {score_dict}")
    plt.suptitle("Pola decyzyjne dla klasyfikatorów wytrenowanych na danych po min-max scalingu")

    fig2 = plt.figure(figsize=(10, 8))
    for classifier, clf_name, grd in zip(classificators, classificators_names, itertools.product([0, 1], repeat=2)):
        score_dict[clf_name] = check_different_classificator(classifier, clf_name,
                                                             x_train_standard_scaled[:, 0:2],
                                                             x_test_standard_scaled[:, 0:2],
                                                             y_train, y_test, gs, grd, fig2)
    print(f"Wyniki klasyfikatorow wytrenowanych nadanych po standard scalingu: {score_dict}")
    plt.suptitle("Pola decyzyjne dla klasyfikatorów wytrenowanych na danych po standard scalingu")
    plt.show()


todo1()
todo2()
