import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn import pipeline

from mlxtend.plotting import plot_decision_regions

from lab_s01_utils import print_function_name


def plot_iris(X: np.ndarray, title: str = 'Iris sepal features') -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title(title)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')


def plot_iris_decision_boundary(X, y, classifier, title: str):
    plt.figure(figsize=(10, 8))
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.5, cmap=ListedColormap(('red', 'green', 'blue'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
      plt.scatter(
          X[y == j, 0], X[y == j, 1], color=ListedColormap(('red', 'green', 'blue'))(i), label=j
      )
    plt.title(title)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend()


def todo_1():
    print_function_name()


def todo_2():
    print_function_name()

    # Załaduj zbiór danych i wyświetl informacje o nim. Przy
    # załadowaniu jako pandas dataframe można skorzystać z metody describe.
    iris = datasets.load_iris(as_frame=True)
    pd.set_option('display.max_columns', None)
    print(iris.frame.describe())


def todo_3_4():
    print_function_name()

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print(f'count y: {np.bincount(y)}')
    # print(y)
    # X = np.append(X, [[50, 1, 1, 1]], axis=0)
    # y = np.append(y, [1])
    # print('\n')
    # print(y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    # Następnie sprawdź, jakie jest procentowe rozłożenie poszczególnych klas
    # w zbiorze treningowym i testowym. Dobrze było by, aby dystrybucje klas
    # próbek w tych zbiorach były identyczne – zmodyfikuj poprzedni kod tak,
    # żeby dane po podziale spełniały ten warunek (wskazówka: słówko stratify).
    print('Default:')
    print(f'count y_train: {np.bincount(y_train)}')
    print(f'count y_test: {np.bincount(y_test)}')

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print('Stratify:')
    print(f'count y_train: {np.bincount(y_train)}')
    print(f'count y_test: {np.bincount(y_test)}')


def todo_5_6():
    print_function_name()

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_min_max_scaled = min_max_scaler.transform(X)

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(X_train)
    X_standard_scaled = standard_scaler.transform(X)

    plot_iris(X, 'no scaling')
    plot_iris(X_min_max_scaled, 'min_max_scaler')
    plot_iris(X_standard_scaled, 'standard_scaler')
    plt.show()


def todo_7_8():
    print_function_name()

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    clf_svm = svm.SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm.fit(
        min_max_scaler.transform(X_train),
        y_train
    )
    acc_svm = metrics.accuracy_score(
        y_test,
        clf_svm.predict(
            min_max_scaler.transform(X_test)
        )
    )
    print(f'acc_svm: {acc_svm}')

    pipe_svm = pipeline.Pipeline(
        [
            ('min_max_scaler', preprocessing.MinMaxScaler()),
            ('clf_svm', svm.SVC(random_state=42, kernel='rbf', probability=True))
        ]
    )
    pipe_svm.fit(X_train, y_train)
    acc_pipe_svm = metrics.accuracy_score(y_test, pipe_svm.predict(X_test))
    print(f'acc_pipe_svm: {acc_pipe_svm}')

    clf_svm_no_scale = svm.SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm_no_scale.fit(X_train, y_train)
    acc_svm_no_scale = metrics.accuracy_score(y_test, clf_svm_no_scale.predict(X_test))
    print(f'acc_svm_no_scale: {acc_svm_no_scale}')

    print('Sample prediction with scale transform:')
    print(clf_svm.predict(min_max_scaler.transform([[8.0, 4.0, 2.0, 2.0]])))
    print(clf_svm.predict(min_max_scaler.transform([[8.0, 50.0, 2.0, 2.0]])))

    print('Sample prediction with probability with scale transform:')
    print(clf_svm.predict_proba(min_max_scaler.transform([[8.0, 4.0, 2.0, 2.0]])))
    print(clf_svm.predict_proba(min_max_scaler.transform([[8.0, 50.0, 2.0, 2.0]])))


def todo_9():
    print_function_name()

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X = X[:, [0, 1]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train_scaled = min_max_scaler.transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)
    clf_svm = svm.SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm.fit(X_train_scaled, y_train)

    pipe_svm = pipeline.Pipeline(
        [
            ('min_max_scaler', preprocessing.MinMaxScaler()),
            ('clf_svm', svm.SVC(random_state=42, kernel='rbf', probability=True))
        ]
    )
    pipe_svm.fit(X_train, y_train)

    clf_svm_no_scale = svm.SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm_no_scale.fit(X_train, y_train)

    plt.figure()
    plot_decision_regions(X_test_scaled, y_test, clf=clf_svm, legend=2)
    plt.figure()
    plot_decision_regions(X_test, y_test, clf=pipe_svm, legend=2)
    plt.figure()
    plot_decision_regions(X_test, y_test, clf=clf_svm_no_scale, legend=2)

    plot_iris_decision_boundary(
        X_test, y_test, pipe_svm, title='Matplotlib based pipe min max scaling classifier decision boundary'
    )
    plt.show()


def main():
    # todo_1() - not implemented yet
    todo_2()
    todo_3_4()
    todo_5_6()
    todo_7_8()
    todo_9()


if __name__ == '__main__':
    main()
