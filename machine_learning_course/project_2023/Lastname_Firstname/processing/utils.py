import numpy as np
import pandas as pd


def perform_processing(
        data: pd.DataFrame
) -> pd.DataFrame:
    # NOTE(MF): sample code
    # preprocessed_data = preprocess_data(data)
    # models = load_models()  # or load one model
    # please note, that the predicted data should be a proper pd.DataFrame with column names
    # predicted_data = predict(models, preprocessed_data)
    # return predicted_data

    # for the simplest approach generate a random DataFrame with proper column names and size
    random_results = np.random.choice(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
         't', 'u', 'v', 'w', 'x', 'y'],
        data.shape[0]
    )
    print(f'{random_results=}')

    predicted_data = pd.DataFrame(
        random_results,
        columns=['letter']
    )

    print(f'{predicted_data=}')

    return predicted_data
