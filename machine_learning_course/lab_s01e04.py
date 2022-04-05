import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from sklearn import datasets
from sklearn import ensemble
from sklearn import impute
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import pipeline, cluster
from sklearn import decomposition, manifold

import gap_statistic

from lab_s01_utils import print_function_name


def todo_1():
    print_function_name()

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    print(X)

    print(X.info())
    print(X.describe())

    # X, y = datasets.fetch_openml('diabetes', return_X_y=True)
    # print(X)
    # print(y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_ii = X_train.copy()

    plt.figure()
    X_train.boxplot()
    X_train.hist()

    # plt.figure()
    # sns.boxplot(x=X_train['mass'])
    # plt.show()

    imputer_mass = impute.SimpleImputer(missing_values=0.0, strategy='mean')
    # print(X[:, 5].reshape(-1, 1))
    # imputer.fit(X[:, 5].reshape(-1, 1))
    # X[:, 5] = imputer.transform(X[:, 5].reshape(-1, 1))
    imputer_skin = impute.SimpleImputer(missing_values=0.0, strategy='mean')

    X_train[['mass']] = imputer_mass.fit_transform(X_train[['mass']]) #.values.reshape(-1, 1))
    X_train[['skin']] = imputer_skin.fit_transform(X_train[['skin']])

    X_test[['mass']] = imputer_mass.transform(X_test[['mass']])
    X_test[['skin']] = imputer_skin.transform(X_test[['skin']])

    # # X_train.boxplot()
    # X_train.hist(bins=20)

    imputer_ii = impute.KNNImputer(n_neighbors=2, missing_values=0.0)

    X_train_ii[['mass']] = imputer_ii.fit_transform(X_train_ii[['mass']])
    X_train_ii[['skin']] = imputer_ii.fit_transform(X_train_ii[['skin']])

    # X_train_2.boxplot()
    # X_train_ii.hist(bins=20)
    # plt.show()

    df_mass = X_train[['mass']]
    print(df_mass.head(5))

    X_train_isolation = X_train.values
    X_train_isolation = X_train_isolation[:, [1, 5]]
    X_test_isolation = X_test.values
    X_test_isolation = X_test_isolation[:, [1, 5]]

    isolation_forest = ensemble.IsolationForest(contamination=0.05)
    isolation_forest.fit(X_train_isolation)
    y_predicted_outliers = isolation_forest.predict(X_test_isolation)
    print(y_predicted_outliers)

    print(X_test_isolation)
    plot_iris(X_test_isolation, y_predicted_outliers)
    plt.show()

    clf_svm = svm.SVC(random_state=42)
    clf_svm.fit(X_train, y_train)
    y_predicted_svm = clf_svm.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_svm))

    clf_rf = ensemble.RandomForestClassifier(random_state=42)
    clf_rf.fit(X_train, y_train)
    y_predicted_rf = clf_rf.predict(X_test)
    print(metrics.classification_report(y_test, y_predicted_rf))

    importances = clf_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    # y[y == 'tested_positive'] = 1
    # y[y == 'tested_negative'] = 0
    # print(y)


def todo_2():
    print_function_name()

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

    plt.figure()
    X.boxplot()
    X.hist(bins=20)
    plt.show()

    plt.figure()
    sns.boxplot(x=X['mass'])
    plt.figure()
    sns.histplot(data=X, x='mass')
    plt.show()


def todo_3():
    print_function_name()

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

    from lab_s01e03 import plot_iris

    plot_iris(X[['mass', 'plas']].to_numpy(), title='mass vs plas')
    plt.show()


def todo_4():
    print_function_name()

    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)
    print(X.describe())

    X_zscore = X.apply(scipy.stats.zscore)
    print(X_zscore.head(5))

    X_filtered = X[X_zscore < 3]
    print(X_filtered.describe())


def main():
    todo_1()  # WIP
    todo_2()
    todo_3()
    todo_4()


if __name__ == '__main__':
    main()
