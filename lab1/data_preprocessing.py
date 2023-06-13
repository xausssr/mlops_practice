import os
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn import preprocessing


def prepare_data(
        base_dir: str, scaler: Union[preprocessing.StandardScaler, None] = None
        ) -> Tuple[pd.DataFrame, preprocessing.StandardScaler]:
    """Предобработка данных (для фич - preprocessing.StandardScaler, для цели - логорифмический масштаб)

    Args:
        base_dir (str): имя входной папки train или test
        scaler (Union[preprocessing.StandardScaler, None]): если данные для теста, подаем предобученный скалер
    Returns:
        Tuple[pd.DataFrame, preprocessing.StandardScaler]: кортеж (готовые данные, скалер)
    """

    chuncks = os.listdir(base_dir)
    data = pd.read_csv(os.path.join(base_dir, chuncks[0]))
    for chunck in chuncks[1:]:
        data = pd.concat([data, pd.read_csv(os.path.join(base_dir, chunck))], axis=0, ignore_index=True)

    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(data.drop(columns=['label']))

    labels = np.log(data['label'].values)
    data = pd.DataFrame(scaler.transform(data.drop(columns=['label'])), columns=data.drop(columns=['label']).columns)
    data['label'] = labels
    return data, scaler


if __name__ == '__main__':
    assert 'train' in os.listdir('./'), 'Нет данных для процессинга (./train)'
    assert 'test' in os.listdir('./'), 'Нет данных для процессинга (./test)'

    data, scaler = prepare_data('./train', scaler=None)
    data.to_csv(os.path.join('./train', 'prepared.csv'), index=False)

    data, scaler = prepare_data('./test', scaler=scaler)
    data.to_csv(os.path.join('./test', 'prepared.csv'), index=False)
