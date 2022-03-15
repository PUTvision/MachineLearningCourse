def classes():
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(digits.data[:-1], digits.target[:-1])

    print(clf.predict(digits.data[-1:]))

    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.33, random_state=42
    )

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    print(clf.predict(X_test))
    print(metrics.classification_report(y_test, clf.predict(X_test)))


def pandas_gui():
    diabetes = datasets.load_diabetes(as_frame=True)
    print(diabetes.frame.head(5))
    from pandasgui import show
    show(diabetes.frame, settings={'block': True})

    print('stop')

def todo_final_boss():
    print('\n\n\nTODO 8')

    data = np.loadtxt(fname='./battery_problem_data.csv', delimiter=',')
    print(data)

    input()

    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    print(y)
    print(len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn import tree
    from sklearn.model_selection import cross_val_score
    regressor = tree.DecisionTreeRegressor()
    regressor.fit(X_train, y_train)

    print(metrics.check_scoring(regressor))

    print(cross_val_score(regressor, X, y, cv=3))

    from sklearn import linear_model
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    print('Coefficients: \n', regr.coef_)

    from sklearn.metrics import mean_squared_error, r2_score
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))

    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))



    y_predicted = regressor.predict(X_test)

    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_predicted))

    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_predicted))

    print(y_predicted)
    print(len(y_predicted))

    print(y_test)
    print(len(y_test))

    print(y_predicted-y_test)

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)

    poly.fit_transform(X_train)

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(X_train, y_train)

    y_poly_predicted = model.predict(X_test)
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_poly_predicted))

    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_poly_predicted))

    plt.scatter(X_test, y_test, marker='o')
    plt.scatter(X_test, y_predicted, marker='x')
    plt.scatter(X_test, y_poly_predicted, color='red', marker='*')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.show()


def regressor_9000(x: float) -> float:
    if x >= 4.0:
        return 8.0
    else:
        # 0.14,0.28
        # 2.00,4.00
        # x     y
        return x*2


def todo_final_boss_2():
    data = np.loadtxt('./battery_problem_data.csv', delimiter=',')
    print(data)

    x = data[:, 0]
    y = data[:, 1]

    y_predicted = []
    for single_data in x:
        y_predicted.append(regressor_9000(single_data))

    plt.scatter(x, y)
    plt.scatter(x, y_predicted, marker='*', c='red')
    plt.show()

from sklearn import datasets, metrics

def make_classification_playground():
    X, y = datasets.make_classification(
        n_samples=10000,
        weights=[0.95, 0.001],
        n_classes=2,
        n_features=4, n_redundant=0, n_repeated=0, n_informative=2,
        class_sep=1.0,
        random_state=1,
        n_clusters_per_class=1
    )
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    from sklearn import svm
    clf = svm.SVC(class_weight='balanced')
    # clf = svm.SVC()
    clf.fit(X_train, y_train)

    print(metrics.classification_report(y_test, clf.predict(X_test)))
    print(metrics.balanced_accuracy_score(y_test, clf.predict(X_test)))
