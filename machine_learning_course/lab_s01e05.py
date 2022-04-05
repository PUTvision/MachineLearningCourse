import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns


from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute


def plot_iris(X: np.ndarray, y: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)





def main():
    todo_1()


if __name__ == '__main__':
    main()
