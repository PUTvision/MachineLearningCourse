import random
import json
import math
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno

from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm


def _plot_one_against_original_data(df_original: pd.DataFrame, df_resampled: pd.DataFrame):
    df_resampled_renamed = df_resampled.rename(columns={'power': 'power_resampled'}, inplace=False)
    ax = df_original.plot()
    df_resampled_renamed.plot(ax=ax)
    # plt.show()


def resample_comparison():
    dates = pd.date_range('20191204', periods=30, freq='5s')
    df = pd.DataFrame(
        {'power': np.random.randint(low=0, high=50, size=len(dates))},
        index=dates
    )

    df_resampled_mean = df.resample('6s').mean()
    df_resampled_nearest = df.resample('6s').nearest()
    df_resampled_ffill = df.resample('6s').ffill()
    df_resampled_bfill = df.resample('6s').bfill()

    df_plt, = plt.plot(df)
    mean_plt, = plt.plot(df_resampled_mean)
    nearest_plt, = plt.plot(df_resampled_nearest)
    bfill_plt, = plt.plot(df_resampled_bfill)
    ffill_plt, = plt.plot(df_resampled_ffill)
    plt.legend(
        [df_plt, mean_plt, nearest_plt, bfill_plt, ffill_plt],
        ['df', 'mean', 'nearest', 'bfill', 'ffill']
    )
    plt.show()

    _plot_one_against_original_data(df, df_resampled_nearest)
    _plot_one_against_original_data(df, df_resampled_mean)
    _plot_one_against_original_data(df, df_resampled_bfill)
    _plot_one_against_original_data(df, df_resampled_ffill)
    plt.show()


def read_temp_sn(name: str) -> int:
    with open('data/additional_info.json') as f:
        additional_data = json.load(f)

    devices = additional_data['offices']['office_1']['devices']
    sn_temp_mid = [d['serialNumber'] for d in devices if d['description'] == name][0]
    return sn_temp_mid


def project_check_data():
    # sn_temp_mid = read_temp_sn('temperature_middle')
    sn_temp_mid = read_temp_sn('radiator_1')

    df_temp = pd.read_csv('data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    print(df_temp.info())
    print(df_temp.describe())
    print(df_temp.head(5))

    df_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_temp.rename(columns={'value': 'temp'}, inplace=True)
    df_temp['time'] = pd.to_datetime(df_temp['time'])
    df_temp.drop(columns=['unit'], inplace=True)
    print(df_temp.info())

    df_temp = df_temp[df_temp['serialNumber'] == sn_temp_mid]
    print(df_temp.info())
    print(df_temp.head(5))

    df_temp.set_index('time', inplace=True)

    df_temp: pd.DataFrame
    # df_temp.plot(kind='scatter')
    plt.scatter(df_temp.index, df_temp.temp)
    plt.show()

    df_target_temp = pd.read_csv('data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_target_temp.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_target_temp.rename(columns={'value': 'target_temp'}, inplace=True)
    df_target_temp['time'] = pd.to_datetime(df_target_temp['time'])
    df_target_temp.drop(columns=['unit'], inplace=True)
    df_target_temp.set_index('time', inplace=True)

    df_valve = pd.read_csv('data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv')
    df_valve.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df_valve.rename(columns={'value': 'valve'}, inplace=True)
    df_valve['time'] = pd.to_datetime(df_valve['time'])
    df_valve.drop(columns=['unit'], inplace=True)
    df_valve.set_index('time', inplace=True)

    df_combined = pd.concat([df_temp, df_target_temp, df_valve])

    df_combined = df_combined.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')
    print(df_combined.head())

    df_combined['temp_last'] = df_combined['temp'].shift(1, fill_value=20)
    df_combined['temp_gt'] = df_combined['temp'].shift(-1, fill_value=20.34)

    mask = (df_combined.index < '2020-10-27')
    df_train = df_combined.loc[mask]

    # X_train = df_train.drop(columns=['target_temp', 'temp_last', 'temp_gt']).to_numpy()[1:-1]
    X_train = df_train[['temp', 'valve']].to_numpy()[1:-1]
    print(X_train[:5])
    y_train = df_train['temp_gt'].to_numpy()[1:-1]

    reg_rf = ensemble.RandomForestRegressor(random_state=42)
    reg_rf.fit(X_train, y_train)

    pickle.dump(reg_rf, open('reg_rf.p', 'wb'))

    mask = (df_combined.index > '2020-10-27') & (df_combined.index <= '2020-10-28')
    df_test = df_combined.loc[mask]

    X_test = df_test[['temp', 'valve']].to_numpy()
    y_predicted = reg_rf.predict(X_test)
    df_test['temp_predicted'] = y_predicted.tolist()

    y_test = df_test['temp_gt'].to_numpy()[1:-1]
    y_last = df_test['temp_last'].to_numpy()[1:-1]
    print(f'mae base: {metrics.mean_absolute_error(y_test, y_last)}')
    print(f'mae rf: {metrics.mean_absolute_error(y_test, y_predicted[1:-1])}')
    print(f'mse base: {metrics.mean_squared_error(y_test, y_last)}')
    print(f'mse rf: {metrics.mean_squared_error(y_test, y_predicted[1:-1])}')

    print(df_combined.head(5))
    print(df_combined.tail(5))
    df_test.drop(columns=['valve', 'temp', 'target_temp'], inplace=True)
    df_test.plot()
    plt.show()

    # df_temp.plot()
    # df_target_temp.plot()
    # plt.plot(df_temp.index, df_temp.temp)
    # plt.plot(df_target_temp.index, df_target_temp.target_temp)
    # plt.show()


def do_magic(
        temperature: pd.DataFrame,
        target_temperature: pd.DataFrame,
        valve_level: pd.DataFrame,
        serial_number_for_prediction: str
) -> float:
    # print(temperature.head(5))
    # print(temperature.tail(5))

    return 20


def preprocess_time_to_index(d: pd.DataFrame) -> pd.DataFrame:
    processed_d = d.rename(columns={'Unnamed: 0': 'time'})
    processed_d['time'] = pd.to_datetime(processed_d['time'])
    return processed_d.set_index('time')


def project_checker():
    np.random.seed(42)

    # start = pd.to_datetime('2020-10-21 8:00', format="%Y-%m-%d %H:%M")
    start = pd.Timestamp('2020-10-21 8:00').tz_localize('UTC')
    stop = pd.Timestamp('2020-10-21 12:00').tz_localize('UTC')

    serial_number_for_prediction = '0015BC0035001299'

    df_temperature = pd.read_csv('data/office_1_temperature_supply_points_data_2020-10-13_2020-11-02.csv')
    df_temperature = preprocess_time_to_index(df_temperature)
    df_target_temperature = pd.read_csv('data/office_1_targetTemperature_supply_points_data_2020-10-13_2020-11-01.csv')
    df_target_temperature = preprocess_time_to_index(df_target_temperature)
    df_valve = pd.read_csv('data/office_1_valveLevel_supply_points_data_2020-10-13_2020-11-01.csv')
    df_valve = preprocess_time_to_index(df_valve)

    df_gt_temperature = df_temperature.resample(pd.Timedelta(minutes=15)).mean().fillna(method='ffill')
    df_gt_temperature['predicted'] = 0
    print(df_gt_temperature.head(5))
    print(df_gt_temperature.tail(5))

    current = start
    while current <= stop:
        df_last_time = df_combined[
            (df_combined.index > current - pd.DateOffset(minutes=15)) & (df_combined.index <= current)]
        df_last_time.sort_index(inplace=True)
        print(current)
        print(df_last_time['temperature'].mean())


        # print(df_last_time.head(5))
        # print(df_last_time.tail(5))

        # print(current)
        # print(type(current))

        predicted_temperature = do_magic(
            df_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_target_temperature.loc[(current - pd.DateOffset(days=7)):current],
            df_valve.loc[(current - pd.DateOffset(days=7)):current],
            serial_number_for_prediction
        )

        # gt_temperature = df_gt_temperature.loc[(current - pd.DateOffset(minutes=1)):(current + pd.DateOffset(minutes=1))]
        gt_temperature = df_gt_temperature['value'].loc[current]
        # gt_temperature = df_gt_temperature[current]
        # print(type(gt_temperature))
        # print(gt_temperature.at['value'])
        error = math.fabs(gt_temperature - predicted_temperature)
        print(error)
        # df_gt_temperature['predicted'].loc[current] = predicted_temperature
        # df_gt_temperature.loc[current, 'predicted'] = predicted_temperature
        # print(df_gt_temperature.loc[current, 'predicted'])
        df_gt_temperature.at[current, 'predicted'] = predicted_temperature
        print(df_gt_temperature.at[current, 'predicted'])
        # print(gt_temperature.head(5))

        current = current + pd.DateOffset(minutes=15)

        print(df_combined_resampled.at[current, 'temperature'])

    df_gt_temperature = df_gt_temperature.loc[start:stop]

    print(df_gt_temperature.head(5))
    print(df_gt_temperature.tail(5))



def main():
    random.seed(42)

    pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    # resample_comparison()

    project_check_data()
    # project_checker()


if __name__ == '__main__':
    main()
