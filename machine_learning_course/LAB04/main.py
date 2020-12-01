from sklearn import datasets
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

iris = datasets.load_iris(as_frame=True)
iris_df = pd.concat([iris.data, iris.target], axis=1)


def todo1():
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.scatter(iris.data.values[:, 0], iris.data.values[:, 1], iris.data.values[:, 2], c=iris.target)

    axs.set_title('Dane wejsciowe')
    axs.set_xlabel('Sepal length (cm)')
    axs.set_ylabel('Sepal width (cm)')
    axs.set_zlabel('Petal width (cm)')

    plt.show()


def todo2():
    k_means = KMeans(n_clusters=3)
    k_means.fit(iris.data)

    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.scatter(iris_df.loc[iris_df["target"] == 0, "sepal length (cm)"],
                iris_df.loc[iris_df["target"] == 0, "sepal width (cm)"],
                iris_df.loc[iris_df["target"] == 0, "petal width (cm)"], color='r')
    axs.scatter(iris_df.loc[iris_df["target"] == 1, "sepal length (cm)"],
                iris_df.loc[iris_df["target"] == 1, "sepal width (cm)"],
                iris_df.loc[iris_df["target"] == 1, "petal width (cm)"], color='g')
    axs.scatter(iris_df.loc[iris_df["target"] == 2, "sepal length (cm)"],
                iris_df.loc[iris_df["target"] == 2, "sepal width (cm)"],
                iris_df.loc[iris_df["target"] == 2, "petal width (cm)"], color='b')

    iris_df["kmean_labels"] = k_means.labels_

    axs.scatter(iris_df.loc[iris_df["kmean_labels"] == 0, "sepal length (cm)"],
                iris_df.loc[iris_df["kmean_labels"] == 0, "sepal width (cm)"],
                iris_df.loc[iris_df["kmean_labels"] == 0, "petal width (cm)"], color='r', marker='+')
    axs.scatter(iris_df.loc[iris_df["kmean_labels"] == 1, "sepal length (cm)"],
                iris_df.loc[iris_df["kmean_labels"] == 1, "sepal width (cm)"],
                iris_df.loc[iris_df["kmean_labels"] == 1, "petal width (cm)"], color='g', marker='+')
    axs.scatter(iris_df.loc[iris_df["kmean_labels"] == 2, "sepal length (cm)"],
                iris_df.loc[iris_df["kmean_labels"] == 2, "sepal width (cm)"],
                iris_df.loc[iris_df["kmean_labels"] == 2, "petal width (cm)"], color='b', marker='+')

    axs.set_title('Przynaleznosc do klas z target')
    axs.set_xlabel('Sepal length (cm)')
    axs.set_ylabel('Sepal width (cm)')
    axs.set_zlabel('Petal width (cm)')
    print("Klasa 0 z target zostala w kmean oznaczona jako klasa 1 i odwrotnie.")

    cluster_centers = k_means.cluster_centers_
    axs.scatter(cluster_centers[0, 0], cluster_centers[0, 1], cluster_centers[0, 3], s=500, color='r', marker='v')
    axs.scatter(cluster_centers[1, 0], cluster_centers[1, 1], cluster_centers[1, 3], s=500, color='g', marker='v')
    axs.scatter(cluster_centers[2, 0], cluster_centers[2, 1], cluster_centers[2, 3], s=500, color='b', marker='v')
    axs.legend(["0 target", "1 target", "2 target", "0 kmean", "1 kmean", "2 kmean", "0 center", "1 center",
                "2 center"])

    print(f"Wynik: {adjusted_rand_score(iris.target.values, k_means.labels_)}")

    plt.show()


def todo3():
    k_means = KMeans(n_clusters=3)
    k_means.fit(iris.data)

    mean_shift = MeanShift()
    mean_shift.fit(iris.data)

    affinity_propagation = AffinityPropagation()
    affinity_propagation.fit(iris.data)

    print(f"Wynik ARS Kmeans: {adjusted_rand_score(iris.target.values, k_means.labels_)}")
    print(f"Wynik ARS Mean shift: {adjusted_rand_score(iris.target.values, mean_shift.labels_)}")
    print(f"Wynik ARS Affinity propagation: {adjusted_rand_score(iris.target.values, affinity_propagation.labels_)}")

    # Czym wyzszy wynik CHI tym lepiej zdefiniowane klastry
    print(f"Wynik CHI Kmeans: {calinski_harabasz_score(iris.data, k_means.labels_)}")
    print(f"Wynik CHI Mean shift: {calinski_harabasz_score(iris.data, mean_shift.labels_)}")
    print(f"Wynik CHI Affinity propagation: {calinski_harabasz_score(iris.data, affinity_propagation.labels_)}")


def todo4():
    n_clusters_list = np.arange(2, 15)

    inertia_dict = {}
    for n_clusters in n_clusters_list:
        k_means = KMeans(n_clusters=n_clusters)
        k_means.fit(iris.data)
        inertia_dict[n_clusters] = k_means.inertia_

    print(inertia_dict)
    inertia_df = pd.DataFrame.from_dict(inertia_dict, columns=["inertia"], orient="index")
    inertia_df.plot(style='--b.')
    plt.show()


todo1()
todo2()
todo3()
todo4()
