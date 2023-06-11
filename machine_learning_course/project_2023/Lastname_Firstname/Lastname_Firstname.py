import argparse
import pathlib

import pandas as pd
from sklearn.metrics import accuracy_score

from processing.utils import perform_processing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('results_file', type=str)
    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file)
    results_file = pathlib.Path(args.results_file)

    data = pd.read_csv(input_file, sep=',')

    output_column_name = ['letter']

    gt_data = data[output_column_name]
    input_data = data.drop(columns=output_column_name)

    predicted_data = perform_processing(input_data)

    # sample for calculating accuracy of the prediction
    print(accuracy_score(gt_data, predicted_data))

    predicted_data.to_csv(results_file, sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
