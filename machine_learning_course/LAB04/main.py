from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd

iris = datasets.load_iris(as_frame=True)
iris_df = pd.concat([iris.data, iris.target], axis=1)


def todo1():
    fig = plt.figure()
    axs = fig.add_subplot(111, projection='3d')
    axs.scatter(iris_df.loc[iris_df["target"] == 0, "sepal length (cm)"],
                iris_df.loc[iris_df["target"] == 0, "sepal width (cm)"],
                iris_df.loc[iris_df["target"] == 0, "petal width (cm)"],    color='r')
    axs.scatter(iris_df.loc[iris_df["target"] == 1, "sepal length (cm)"],
                iris_df.loc[iris_df["target"] == 1, "sepal width (cm)"],
                iris_df.loc[iris_df["target"] == 1, "petal width (cm)"],    color='g')
    axs.scatter(iris_df.loc[iris_df["target"] == 2, "sepal length (cm)"],
                iris_df.loc[iris_df["target"] == 2, "sepal width (cm)"],
                iris_df.loc[iris_df["target"] == 2, "petal width (cm)"],    color='b')

    axs.legend([iris["target_names"][0], iris["target_names"][1], iris["target_names"][2]])

    axs.set_title('Dane wejsciowe')
    axs.set_xlabel('Sepal length (cm)')
    axs.set_ylabel('Sepal width (cm)')
    axs.set_zlabel('Petal width (cm)')

    plt.show()


todo1()
