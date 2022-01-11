import pandas as pd
import darts
import matplotlib.pyplot as plt

from darts.models import ExponentialSmoothing, NaiveDrift, NaiveMean, AutoARIMA, RegressionModel
from darts.metrics import mape
import sklearn


def todo_1():
    print('pip install "u8darts[torch, pmdarima]"')


def todo_2():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    print(df.describe())

    df.plot()
    plt.show()

    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')
    print(series)
    series.plot()
    plt.show()


def todo_3():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train1, test1 = series[:-36], series[-36:]
    train2, test2 = series.split_before(pd.Timestamp('19580101'))

    train1.plot()
    test1.plot()
    plt.show()

    train2.plot()
    test2.plot()
    plt.show()


def todo_4_5():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=",")
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train, test = series[:-36], series[-36:]

    model = NaiveDrift()
    model.fit(train)
    prediction = model.predict(len(test))

    score = mape(actual_series=test, pred_series=prediction)
    print(f'{score=}')

    train.plot()
    test.plot(label='test')
    prediction.plot(label='forecast')
    plt.legend()
    plt.show()


def main():
    todo_1()
    todo_2()
    todo_3()
    todo_4_5()


if __name__ == '__main__':
    main()
