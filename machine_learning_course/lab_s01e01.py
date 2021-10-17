from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np


def classes():
    digits = datasets.load_digits()

    print(digits.DESCR)
    print(f'digits data:\n {digits.data},\n len: {len(digits.data)}')
    print(f'digits target:\n {digits.target},\n len: {len(digits.target)}')
    print(f'digits target_names:\n {digits.target_names}')
    print(f'digits images:\n {digits.images},\n len: {len(digits.images)}')

    print(digits.data[-1])
    print(digits.images[-1])

    import matplotlib.pyplot as plt
    #plt.figure()
    index = -1
    print(digits.target[index])
    plt.imshow(digits.images[index], cmap=plt.cm.gray_r)
    plt.show()

    plt.imshow([digits.data[index]], cmap=plt.cm.gray_r)
    plt.show()

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


def todo_4():
    faces = datasets.fetch_olivetti_faces()

    image_shape = (64, 64)
    n_row, n_col = 2, 3
    n_components = n_row * n_col

    def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
        plt.figure(figsize=(2. * n_col, 2.26 * n_row))
        plt.suptitle(title, size=16)
        for i, comp in enumerate(images):
            plt.subplot(n_row, n_col, i + 1)
            vmax = max(comp.max(), -comp.min())
            plt.imshow(comp.reshape(image_shape), cmap=cmap,
                       interpolation='nearest',
                       vmin=-vmax, vmax=vmax)
            plt.xticks(())
            plt.yticks(())
        plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    print(f'target: {faces.target[:n_components]}')

    plot_gallery('Olivetti faces', faces.images[:n_components])
    plt.show()


def todo_5():
    diabetes = datasets.load_diabetes(as_frame=True)
    print(diabetes)
    print(diabetes.DESCR)
    print(f'diabetes data:\n {diabetes.data},\n len: {len(diabetes.data)}')
    print(f'diabetes target:\n {diabetes.target},\n len: {len(diabetes.target)}')
    print(f'diabetes feature_names:\n {diabetes.feature_names}')
    print(diabetes.frame.head(5))
    print(diabetes.frame.info())


def todo_6():
    X, y = datasets.make_classification(
        n_features=3, n_redundant=0, n_repeated=0, n_informative=3,
        n_classes=3,
        class_sep=3.0,
        random_state=1,
        n_clusters_per_class=2
    )
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.show()

    print(y)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=25, c=y)
    plt.show()


def todo_7():
    pass


def todo_final_boss():
    data = np.loadtxt(fname='./battery_problem_data.csv', delimiter=',')
    # print(data)

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


def main():
    classes()
    todo_4()
    todo_5()
    todo_6()
    todo_7()
    todo_final_boss()

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


if __name__ == '__main__':
    main()
