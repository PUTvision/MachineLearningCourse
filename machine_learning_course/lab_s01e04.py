import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import pipeline, cluster
from sklearn import decomposition, manifold

import gap_statistic


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





def main():
    # todo_1()
    todo_2()


if __name__ == '__main__':
    main()
