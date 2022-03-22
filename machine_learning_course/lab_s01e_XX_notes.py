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
