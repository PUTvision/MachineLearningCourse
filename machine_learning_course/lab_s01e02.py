import matplotlib.pyplot as plt
import numpy as np

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import pipeline


def todo1():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict(
        [
            [1, 1],
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, -1]
        ]
    ))

    tree.plot_tree(
        clf,
        feature_names=['X1', 'X2'],
        # class_names=[],
        filled=True
    )
    plt.show()


def todo2():
    dict_brand = {'VW': 0, 'Ford': 1, 'Opel': 2}
    dict_damaged = {'tak': 0, 'nie': 1}
    print(dict_brand['VW'])

    # marka - przebieg - czy uszkodzony
    X = [
        ['Opel', 250000, 'tak'],
        ['Opel',  50000, 'nie'],
        ['Opel', 100000, 'tak'],
        ['Ford', 300000, 'nie'],
        ['VW',     5000, 'tak'],
        ['VW',   400000, 'nie'],
        ['Ford',  75000, 'nie']
    ]

    for x in X:
        x[0] = dict_brand[x[0]]
        x[2] = dict_damaged[x[2]]
    print(X)

    y = [
        0,
        1,
        0,
        1,
        1,
        0,
        1
    ]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict(
        [
            [dict_brand['Opel'], 100000, dict_damaged['nie']]
        ]
    ))

    tree.plot_tree(
        clf,
        feature_names=['marka', 'przebieg', 'nie'],
        class_names=['zrezygnować', 'kupić'],
        filled=True
    )
    plt.show()


def todo3():
    digits = datasets.load_digits()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        digits.data, digits.target, test_size=0.33, random_state=42
    )

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    print(f'Classification report: \n{metrics.classification_report(y_test, y_predicted)}')
    print(f'Confusion matrix: \n{metrics.confusion_matrix(y_test, y_predicted)}')

    metrics.plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

    for digit, gt, pred in zip(X_test, y_test, y_predicted):
        if gt != pred and (gt == 3 or gt == 8):
            print(f'Sample {digit} classified as {pred} while it should be {gt}')
            plt.imshow(digit.reshape(8, 8), cmap=plt.cm.gray_r)
            plt.show()


def print_regressor_score(y_test: np.ndarray, y_predicted: np.ndarray) -> None:
    print(f'mae: {metrics.mean_absolute_error(y_test, y_predicted)}')
    print(f'mse: {metrics.mean_squared_error(y_test, y_predicted)}')
    print(f'r2: {metrics.r2_score(y_test, y_predicted)}')


def todo4():
    data = np.loadtxt(fname='./battery_problem_data.csv', delimiter=',')

    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    decision_tree_regressor = tree.DecisionTreeRegressor()
    decision_tree_regressor.fit(X_train, y_train)
    y_predicted_decision_tree = decision_tree_regressor.predict(X_test)

    linear_model_regressor = LinearRegression()
    linear_model_regressor.fit(X_train, y_train)
    y_predicted_linear = linear_model_regressor.predict(X_test)

    polynomial_regressor = pipeline.Pipeline(
        [
            ('poly', preprocessing.PolynomialFeatures(degree=9)),
            ('linear', LinearRegression())
        ]
    )
    polynomial_regressor.fit(X_train, y_train)
    y_predicted_polynomial = polynomial_regressor.predict(X_test)

    print('Decision tree regressor:')
    print_regressor_score(y_test, y_predicted_decision_tree)
    print('Linear regressor:')
    print_regressor_score(y_test, y_predicted_linear)
    print(f'coeff: {linear_model_regressor.coef_}')
    print('Polynomial regressor:')
    print_regressor_score(y_test, y_predicted_polynomial)

    plt.scatter(X_test, y_predicted_decision_tree, c='green', marker='o')
    plt.scatter(X_test, y_test, c='red', marker='*')
    plt.scatter(X_test, y_predicted_linear, c='orange', marker='+')
    plt.plot(X_test, y_predicted_linear, c='orange')
    plt.scatter(X_test, y_predicted_polynomial, c='blue', marker='.')
    plt.show()


def main():
    todo1()
    todo2()
    todo3()
    todo4()


if __name__ == '__main__':
    main()
