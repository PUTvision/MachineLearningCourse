from common import read_csv_file

from sklearn import datasets, svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from matplotlib import pyplot as plt


def todo1_and():
    # Wiersze - próbki uczące
    # Kolumny - cechy. Kazda probka uczaca ma 2 cechy
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 0, 0, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    plot_tree(clf)
    plt.show()
    print(clf.predict([[1, 1]]))


def todo2_or():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    print(clf.predict([[1, 1]]))
    plot_tree(clf)
    plt.show()


def todo3():
    marki = {"Mazda": 0, "Hyundai": 1, "Honda": 2, "VW": 3, "Ford": 4, "Opel": 5, "Fiat": 6, "Ferrari": 7}

    X = [[marki["Mazda"], 40000, True],
         [marki["Mazda"], 125000, False],
         [marki["Hyundai"], 125000, False],
         [marki["VW"], 100000, True],
         [marki["Honda"], 90000, False],
         [marki["Ferrari"], 250000, False],
         [marki["Ford"], 200000, True],
         [marki["Opel"], 50000, False],
         [marki["Fiat"], 5000, False]]
    y = [True, True, True, False, True, False, False, True, False]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    print(clf.predict([[marki["Fiat"], 2000, False]]))
    plot_tree(clf)
    plt.show()


def todo4():
    digits = datasets.load_digits()
    data = digits['data']
    targets = digits['target']

    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.8, shuffle=True)
    svc_clf_normal = svm.SVC()
    svc_clf_normal.fit(x_train, y_train)
    prediction = svc_clf_normal.predict(x_test)

    print(f"Confusion matrix:\n {confusion_matrix(y_test, prediction)}")


def todo5():
    """
    wyznacz metryki (https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
    np. mean absolute error, mean squared error, r2 dla LinearRegression oraz DecisionTreeRegressor.
    :return:
    """
    column_names = ['charging_time', 'battery_lasted']
    status, data = read_csv_file('../LAB01/trainingdata.txt', column_names)

    if not status:
        return

    train_data, test_data = train_test_split(data.values, test_size=0.8, shuffle=True)

    reg = LinearRegression()
    reg.fit(train_data[:, 0].reshape(-1, 1), train_data[:, 1])
    score = reg.score(test_data[:, 0].reshape(-1, 1), test_data[:, 1])
    print("Wynik modelu: ", score)
    prediciton = reg.predict(test_data[:, 0].reshape(-1, 1))
    print(f"Mean squared error: {mean_squared_error(test_data[:, 1], prediciton)}")
    print(f"Mean absolute error: {mean_absolute_error(test_data[:, 1], prediciton)}")
    plt.scatter(test_data[:, 0].reshape(-1, 1), test_data[:, 1].reshape(-1, 1), label="Rzeczywiste")
    plt.scatter(test_data[:, 0].reshape(-1, 1), prediciton, label="Predykcja")
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.legend()
    plt.show()

    print("\n")

    reg = DecisionTreeRegressor()
    reg.fit(train_data[:, 0].reshape(-1, 1), train_data[:, 1])
    score = reg.score(test_data[:, 0].reshape(-1, 1), test_data[:, 1])
    print("Decision Tree Regressor: ")
    print("Wynik modelu: ", score)
    prediciton = reg.predict(test_data[:, 0].reshape(-1, 1))
    print(f"Mean squared error: {mean_squared_error(test_data[:, 1], prediciton)}")
    print(f"Mean absolute error: {mean_absolute_error(test_data[:, 1], prediciton)}")

    plt.scatter(test_data[:, 0].reshape(-1, 1), test_data[:, 1].reshape(-1, 1), label="Rzeczywiste")
    plt.scatter(test_data[:, 0].reshape(-1, 1), prediciton, label="Predykcja")
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.legend()
    plt.show()


# todo1_and()
# todo2_or()
# todo3()
# todo4()
todo5()
