import os
import string
import numpy as np
import pandas as pd


def create_moc_data() -> pd.DataFrame:
    """Создание тестового датасета

    Returns:
        pd.DataFrame: датасет размерностью 1000х26
    """

    return pd.DataFrame(
        data=np.random.randn(1000, 26),
        columns=list(string.ascii_lowercase)
    )


if __name__ == '__main__':

    df = create_moc_data()
    if 'data' not in os.listdir('./'):
        os.mkdir('data')
    df.to_csv("data/moc_data.csv")
