import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno

import random

from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm


def random_cabin(clas):
    if clas == 1.0:
        deck = random.choices(['C', 'B', 'D', 'E', 'A', 'T'],
                              weights=[0.346405, 0.241830, 0.169935, 0.130719, 0.104575, 0.006536])
        # probability taken from occurance of deck in clas

    elif clas == 2.0:
        deck = random.choices(['D', 'E', 'F'], weights=[0.333333, 0.222222, 0.444444])
    else:
        deck = random.choices(['E', 'F', 'G'], weights=[0.125, 0.750, 0.125])

    number = random.randint(1, 130)

    return deck[0] + str(number)


def fill_cabin(d):
    m = d['cabin'].isnull()
    l = m.sum()
    fills = []
    clas = list(d['ticket'])[0]

    for _ in range(0, l):
        cabin = random_cabin(clas)

        fills.append(cabin)
    d.loc[m, 'cabin'] = fills
    return d


def function(d):
    d_temp = d.copy()
    d_temp['age'] = d_temp['age'].dropna()
    # d_temp['age'].hist()
    # plt.show()

    m = d['age'].isnull()
    l = m.sum()

    s = np.random.normal(d_temp['age'].mean(), d_temp['age'].std(), l)

    s = [item if item >= 0. else np.random.ranf() for item in s]

    d.loc[m, 'age'] = s

    return d


def histogram_1(X_comb):
    survived = 'survived'
    not_survived = 'not survived'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    women = X_comb[X_comb['sex'] == 'female']
    men = X_comb[X_comb['sex'] == 'male']
    ax = sns.distplot(women[women['survived'] == 1.0].age.dropna(), bins=10, label=survived, ax=axes[0], kde=False)
    ax = sns.distplot(women[women['survived'] == 0.0].age.dropna(), bins=10, label=not_survived, ax=axes[0], kde=False)
    ax.legend()
    ax.set_title('Female')
    ax = sns.distplot(men[men['survived'] == 1.0].age.dropna(), bins=10, label=survived, ax=axes[1], kde=False)
    ax = sns.distplot(men[men['survived'] == 0.0].age.dropna(), bins=10, label=not_survived, ax=axes[1], kde=False)
    ax.legend()
    _ = ax.set_title('Male')
    plt.show()


def todo():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    random.seed(42)

    X, y = datasets.fetch_openml(name='Titanic', version=1, return_X_y=True, as_frame=True)
    X: pd.DataFrame = X
    y: pd.DataFrame = y

    print(X.head(5))
    print(X.info())
    print(X.describe())

    print(y.head(5))

    X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)
    print(X.head(5))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    print(X_train.info())

    y_predict_random = random.choices(['0', '1'], k=len(y_test))
    print(metrics.classification_report(y_test, y_predict_random))

    y_predict_0 = ['0']*len(y_test)
    print(metrics.classification_report(y_test, y_predict_0))

    print(f'y.value_counts(): \n{y.value_counts()}')
    print(f'y.value_counts(): \n{y_train.value_counts()}')
    print(f'y.value_counts(): \n{y_test.value_counts()}')

    # msno.matrix(X)
    # plt.show()

    fill_cabin(X_train)
    X_train = X_train.groupby('pclass').apply(function)

    X_combined = pd.concat([X_train, y_train.astype(float)], axis=1)
    print(X_combined.head(5))

    df_temp = X_combined[['sex', 'survived']].groupby('sex').mean()
    print(df_temp.head(5))
    df_temp = X_combined[['pclass', 'survived']].groupby('pclass').mean()
    print(df_temp.head(5))

    histogram_1(X_combined)

    X_combined['sex'].replace({'male': 0, 'female': 1}, inplace=True)
    X_combined.boxplot()
    plt.show()

    print(X_combined.corr())
    g = sns.heatmap(X_combined.corr(),
                    annot=True,
                    cmap="coolwarm")
    plt.show()








def main():
    todo()


if __name__ == '__main__':
    main()
