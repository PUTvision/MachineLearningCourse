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


def todo_20():
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


def plot_iris(X: np.ndarray, y: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma')
    plt.axvline(x=0)
    plt.axhline(y=0)


def plot_iris_3d(X: np.ndarray, y: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)


def todo_1():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    # X = X[:, [0, 1]]

    # zastosuj elbow method dla algorytmu k-means w tym celu:
    # w pętlil wytrenuj algorytm k-means dla liczb klastrów od 2 do 15
    # dla każdej iteracji zapisz atrybut interia_
    # wygeneruj wykres zapisanych wartości
    # dlaczego wykres jest "odwrócony" względem przykładu z Wikipedii? Czy to poprawne?
    # dobierz liczbę klastrów? Czy łatwą ją określić?
    # wykorzystaj gap statistics zaimplementowaną w pakiecie gap_stat. Zobrazuj wyznaczane tam wartości za pomocą dostępnej metody plot_results.

    optimalK = gap_statistic.OptimalK(n_jobs=4, parallel_backend='joblib')
    n_clusters = optimalK(X, cluster_array=np.arange(2, 15))
    print(n_clusters)
    print(optimalK.gap_df.head())
    gap_statistic.OptimalK.plot_results(optimalK)

    clusters = range(2, 15)
    inertias = []
    for n in clusters:
        kmeans = cluster.KMeans(n_clusters=n).fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(clusters, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(clusters)
    plt.show()

    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(X)
    kmeans_3 = kmeans.labels_
    print(kmeans_3)
    print(y)

    kmeans = cluster.KMeans()
    kmeans.fit(X)
    kmeans_default = kmeans.predict(X)
    print(kmeans_default)

    db = cluster.DBSCAN().fit(X)
    db_labels = db.labels_

    affinity = cluster.AffinityPropagation().fit(X)
    affinity_labels = affinity.labels_
    print(np.unique(affinity_labels))

    clustering = cluster.AgglomerativeClustering(n_clusters=8).fit(X)
    clustering_labels = clustering.labels_

    print(kmeans.predict([[8.0, 4.0, 0.0, 0.0]]))
    print(kmeans.predict([[8.0, 50.0, 0.0, 0.0]]))

    print('adjusted_rand_score:')
    print(f'kmeans3: {metrics.adjusted_rand_score(y, kmeans_3)}')
    print(f'kmeans_default: {metrics.adjusted_rand_score(y, kmeans_default)}')
    print(f'db_labels: {metrics.adjusted_rand_score(y, db_labels)}')
    print(f'clustering_labels: {metrics.adjusted_rand_score(y, clustering_labels)}')
    print(f'affinity_labels: {metrics.adjusted_rand_score(y, affinity_labels)}')

    print('calinski_harabasz_score:')
    print(f'kmeans3: {metrics.calinski_harabasz_score(X, kmeans_3)}')
    print(f'kmeans_default: {metrics.calinski_harabasz_score(X, kmeans_default)}')
    print(f'db_labels: {metrics.calinski_harabasz_score(X, db_labels)}')
    print(f'clustering_labels: {metrics.calinski_harabasz_score(X, clustering_labels)}')
    print(f'affinity_labels: {metrics.calinski_harabasz_score(X, affinity_labels)}')

    plot_iris_3d(X, y)
    plot_iris_3d(X, kmeans_3)
    plot_iris_3d(X, kmeans_default)
    plot_iris_3d(X, db_labels)
    plot_iris_3d(X, clustering_labels)
    plot_iris_3d(X, affinity_labels)
    plt.show()


def todo_2():
    iris = datasets.load_digits()

    X, y = iris.data, iris.target

    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    tsne = manifold.TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    plot_iris(X, y)
    plot_iris(X_pca, y)
    plot_iris(X_tsne, y)
    plt.show()

def todo_3():
    # Spróbujmy napisać narzędzie do automatycznego sortowania zdjęć z wakacji – używając uczenia nienadzorowanego. Na początek pobierzmy po 20 zdjęć z zapytania ‘beach’ i ‘forest’ z grafik Google.
    # Co najlepiej wykorzystać w charakterze cech? Dwie kategorie powinny różnić się znacząco histogramami kolorów - na zdjęciach plaży dominować będzie kolor nieba i piasku, a na zdjęciach lasu będą to odcienie zieleni (o ile las nie będzie zasłonięty przez drzewa).
    # Po obliczeniu histogramów kolorów dla trzech kanałów (niech mają po 8 do 16 kubełków każdy) powinniśmy mieć dość cech, aby nauczyć nasz algorytm klastrowania. Oczywiście cechy muszą być wyznaczone dla wszystkich obrazów ze zbioru uczącego.
    # Wykorzystaj poznane metody redukcji wymiarowości do zwizualizowania danych.
    # Po nauczeniu, podobnie jak w przypadku klasyfikatorów, możemy za pomocą metody predict podsunąć naszemu obiektowi klastrującemu zupełnie nowe zdjęcia, których nie widział wcześniej – powinien on sobie poradzić z ich zaszeregowaniem.
    # Wskazówka: Histogramy obrazów można obliczyć w OpenCV następującą funkcją:
    # cv2.calcHist(images, channels, mask, histSize, ranges)
    # przykładowo:
    # hist = cv2.calcHist([obraz], [1], None, [8], [0,256])
    # Gdzie [obraz] oznacza nasz obrazek, [1] oznacza kanał z kolorem zielonym (kolejność kanałów w OpenCV to B, G, R), a [8] to docelowa liczba kubełków histogramu.
    pass