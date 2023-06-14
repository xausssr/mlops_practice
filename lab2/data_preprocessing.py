import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


def prepare_data() -> pd.DataFrame:
    """Подготовка датасета для обучения и валидации

    Returns:
        pd.DataFrame: подготовленный датасет
    """

    data = pd.read_csv(os.path.join('data', 'train.csv')).drop(columns='id').iloc[:1000]
    target = 'glasses'

    embed = TSNE(
        n_components=8, learning_rate='auto', init='random', perplexity=35, method='exact'
    ).fit_transform(data.drop(columns=[target]).values)

    return train_test_split(embed, data[target].values, train_size=0.9)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data()
    np.save(os.path.join('data', 'train_x'), X_train)
    np.save(os.path.join('data', 'train_y'), y_train)
    np.save(os.path.join('data', 'test_x'), X_test)
    np.save(os.path.join('data', 'test_y'), y_test)
