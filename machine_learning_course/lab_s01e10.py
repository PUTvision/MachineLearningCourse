from typing import Tuple

import matplotlib.pyplot as plt
import sklearn
import pandas as pd

import darts
from darts.models import ExponentialSmoothing, NaiveDrift, NaiveMean, AutoARIMA, RegressionModel, NaiveSeasonal, Theta
from darts.metrics import mape
from darts.utils.statistics import check_seasonality


def load_and_split_air_passenger() -> Tuple[darts.TimeSeries, darts.TimeSeries]:
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=',')
    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')

    train, test = series[:-36], series[-36:]

    return train, test


def todo_1():
    print('pip install "u8darts[torch, pmdarima]"')


def todo_2():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=',')
    print(df.describe())
    print(df.head(5))
    print(df.dtypes)

    df.plot()
    # plt.show()

    series = darts.TimeSeries.from_dataframe(df, 'Month', '#Passengers')
    print(series)
    plt.figure()
    series.plot()
    plt.show()


def todo_3():
    df = pd.read_csv('.\..\data\AirPassengers.csv', delimiter=',')
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
    train, test = load_and_split_air_passenger()

    model = NaiveDrift()
    model.fit(train)
    prediction = model.predict(len(test))

    mape_score = mape(actual_series=test, pred_series=prediction)
    print(f'{mape_score=} for {model}')

    train.plot()
    test.plot(label='test')
    prediction.plot(label='forecast')
    plt.legend()
    plt.title(f'{model}')
    plt.show()


def todo_6():
    train, test = load_and_split_air_passenger()

    for m in range(2, 48):
        is_seasonal, period = check_seasonality(train, m=m, max_lag=48, alpha=.05)
        if is_seasonal:
            print(f'There is seasonality of order {period}.')

    model = NaiveSeasonal(K=12)
    model.fit(train)
    prediction = model.predict(len(test))

    mape_score = mape(actual_series=test, pred_series=prediction)
    print(f'{mape_score=} for {model}')

    train.plot()
    test.plot(label='test')
    prediction.plot(label='forecast')
    plt.legend()
    plt.title(f'{model}')
    plt.show()


def todo_6_last_question():
    train, test = load_and_split_air_passenger()

    model_drift = NaiveDrift()
    model_drift.fit(train)
    model_seasonal = NaiveSeasonal(K=12)
    model_seasonal.fit(train)

    prediction_drift = model_drift.predict(len(test))
    prediction_seasonal = model_seasonal.predict(len(test))

    prediction_combined = prediction_drift + prediction_seasonal - train.last_value()
    mape_score = mape(actual_series=test, pred_series=prediction_combined)
    print(f'{mape_score=} for combined {model_drift} and {model_seasonal}')

    train.plot()
    test.plot(label='test')
    prediction_combined.plot(label='prediction_combined')
    prediction_drift.plot(label='prediction_drift')
    prediction_seasonal.plot(label='prediction_seasonal')
    plt.legend()
    plt.title(f'{model_drift} + {model_seasonal}')
    plt.show()


def todo_7():
    train, test = load_and_split_air_passenger()

    model_exponential = ExponentialSmoothing()
    model_exponential.fit(train)

    model_theta = Theta()
    model_theta.fit(train)

    prediction_exponential = model_exponential.predict(len(test))
    mape_score = mape(actual_series=test, pred_series=prediction_exponential)
    print(f'{mape_score=} for {model_exponential}')

    prediction_theta = model_theta.predict(len(test))
    mape_score = mape(actual_series=test, pred_series=prediction_theta)
    print(f'{mape_score=} for {model_theta}')

    train.plot()
    test.plot(label='test')
    prediction_exponential.plot(label='prediction_exponential')
    prediction_theta.plot(label='prediction_theta')
    plt.legend()
    plt.show()


def todo_8():
    train, test = load_and_split_air_passenger()

    model = RegressionModel(60, model=sklearn.ensemble.GradientBoostingRegressor())
    model.fit(train)

    prediction = model.predict(len(test))
    mape_score = mape(actual_series=test, pred_series=prediction)
    print(f'{mape_score=} for {model}')

    train.plot()
    test.plot(label='test')
    prediction.plot(label='forecast')
    plt.legend()
    plt.title(f'{model}')
    plt.show()


def todo_9():
    # TODO(MF): add support for another csv with data
    # TODO(MF): add sample with loading dataset from darts library
    darts.datase
    df = pd.read_csv('.\..\data\daily_min_temperatures_Melbourne_1981_1990.csv', delimiter=",")
    pass


def main():
    todo_1()
    todo_2()
    todo_3()
    todo_4_5()
    todo_6()
    todo_6_last_question()
    todo_7()
    todo_8()


if __name__ == '__main__':
    main()
