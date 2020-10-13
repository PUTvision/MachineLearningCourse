import pandas as pd


def read_csv_file(path_to_file: str, names: list):
    try:
        data_frame = pd.read_csv(path_to_file, names=names)
        # print(data_frame)
        return True, data_frame
    except FileNotFoundError as e:
        print(e)
        return False, None
