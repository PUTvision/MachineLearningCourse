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


def plot_iris(X: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')


def todo1():
    # Załaduj zbiór danych i wyświetl informacje o nim. Przy załadowaniu jako pandas dataframe można skorzystać z metody describe.
    iris = datasets.load_iris(as_frame=True)
    print(iris.frame.describe())

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # print(y)
    # X = np.append(X, [[50, 1, 1, 1]], axis=0)
    # y = np.append(y, [1])
    # print('\n')
    # print(y)

    # Następnie sprawdź, jakie jest procentowe rozłożenie poszczególnych klas w zbiorze treningowym i testowym.
    # Dobrze było by, aby dystrybucje klas próbek w tych zbiorach były identyczne – zmodyfikuj poprzedni kod tak,
    # żeby dane po podziale spełniały ten warunek (wskazówka: słówko stratify).

    print(f'count y: {np.bincount(y)}')

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y)
    print(f'count y_train: {np.bincount(y_train)}')
    print(f'count y_test: {np.bincount(y_test)}')

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_min_max_scaled = min_max_scaler.transform(X)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_standard_scaled = scaler.transform(X)

    plot_iris(X)
    plot_iris(X_min_max_scaled)
    plot_iris(X_standard_scaled)
    plt.show()


def plot_iris_decision_boundary(X, y, classifier):
    plt.figure(figsize=(10, 8))
    X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
                      np.arange(start = X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
              alpha=0.5, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
      plt.scatter(X[y == j, 0], X[y == j, 1],
                  color=ListedColormap(('red', 'green', 'blue'))(i), label=j)
    plt.title('Classifier decision boundary')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend()



def todo_2():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print(X)
    X = X[:, [0, 1]]
    print(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)

    X_train_scaled = min_max_scaler.transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)

    pipe_svm = pipeline.Pipeline([('min_max_scaler', preprocessing.MinMaxScaler()), ('clf_svm', svm.SVC(random_state=42, kernel='rbf', probability=True))])
    pipe_svm.fit(X_train, y_train)
    acc_pipe_svm = metrics.accuracy_score(y_test, pipe_svm.predict(X_test))
    print(f'acc_pipe_svm: {acc_pipe_svm}')

    clf_svm_no_scale = svm.SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm_no_scale.fit(X_train, y_train)
    acc_svm_no_scale = metrics.accuracy_score(y_test, clf_svm_no_scale.predict(X_test))
    print(f'acc_svm_no_scale: {acc_svm_no_scale}')

    clf_svm = svm.SVC(random_state=42, kernel='rbf', probability=True)
    clf_svm.fit(X_train_scaled, y_train)
    acc_svm = metrics.accuracy_score(y_test, clf_svm.predict(X_test_scaled))
    print(f'acc_svm: {acc_svm}')

    # clf_linear = linear_model.LogisticRegression(random_state=42)
    # clf_linear.fit(X_train_scaled, y_train)
    # acc_liner = metrics.accuracy_score(y_test, clf_linear.predict(X_test_scaled))
    # print(f'acc_liner: {acc_liner}')
    #
    # clf_tree = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
    # clf_tree.fit(X_train_scaled, y_train)
    # acc_tree = metrics.accuracy_score(y_test, clf_tree.predict(X_test_scaled))
    # print(f'acc_tree: {acc_tree}')
    #
    # clf_rf = ensemble.RandomForestClassifier(random_state=42, n_estimators=10)
    # clf_rf.fit(X_train_scaled, y_train)
    # acc_rf = metrics.accuracy_score(y_test, clf_rf.predict(X_test_scaled))
    # print(f'acc_rf: {acc_rf}')

    # TODO(MF): sprawdzić dlaczego inna przestrzeń decyzyjna z i bez skalowania

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # clf_gs = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, n_jobs=20, verbose=20)
    # clf_gs.fit(X_train, y_train)
    # print(clf_gs.cv_results_)

    print(clf_svm.predict(min_max_scaler.transform([[8.0, 4.0]])))
    print(clf_svm.predict(min_max_scaler.transform([[8.0, 50.0]])))

    print(clf_svm.predict_proba(min_max_scaler.transform([[8.0, 4.0]])))
    print(clf_svm.predict_proba(min_max_scaler.transform([[8.0, 50.0]])))

    # print(clf_svm.predict([[8.0, 4.0]]))
    # print(clf_svm.predict([[8.0, 50.0]]))

    # plot_iris_decision_boundary(X_train, y_train, clf_svm)
    plt.figure()
    plot_decision_regions(X_test, y_test, clf=pipe_svm, legend=2)
    plt.figure()
    plot_decision_regions(X_test, y_test, clf=clf_svm_no_scale, legend=2)
    plt.figure()
    plot_decision_regions(X_test_scaled, y_test, clf=clf_svm, legend=2)
    # plt.figure()
    # plot_decision_regions(X_test_scaled, y_test, clf=clf_linear, legend=2)
    # plt.figure()
    # plot_decision_regions(X_test_scaled, y_test, clf=clf_tree, legend=2)
    # plt.figure()
    # plot_decision_regions(X_test_scaled, y_test, clf=clf_rf, legend=2)
    plt.show()


def main():
    # todo1()
    todo_2()


if __name__ == '__main__':
    main()
