import inspect

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def print_function_name(number_of_empty_lines_before: int = 3) -> None:
    """Print the name of the calling function.

    Parameters
    ----------
    number_of_empty_lines_before : int, optional
        Number of empty lines to print before the function name. Default is 3.
    """
    print('\n'*number_of_empty_lines_before)
    print(f'{inspect.stack()[1][3]}')


def plot_iris(X: np.ndarray, title: str = 'Iris sepal features') -> None:
    """Plot a 2D scatter plot of the first two features of the Iris dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    title : str, optional
        Title for the plot. Default is 'Iris sepal features'.
    """
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title(title)
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')


def plot_iris_decision_boundary(X: np.ndarray, y: np.ndarray, classifier, title: str) -> None:
    """Plot the decision boundary of a classifier on the first two features of the Iris dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    classifier : object
        Trained classifier with a predict method.
    title : str
        Title for the plot.
    """
    plt.figure(figsize=(10, 8))
    X1, X2 = np.meshgrid(
        np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    )
    plt.contourf(
        X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.5, cmap=ListedColormap(('red', 'green', 'blue'))
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
      plt.scatter(
          X[y == j, 0], X[y == j, 1], color=ListedColormap(('red', 'green', 'blue'))(i), label=j
      )
    plt.title(title)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend()
