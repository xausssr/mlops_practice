import os
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


if __name__ == '__main__':
    assert 'prepared.csv' in os.listdir('./train'), 'Нет предобработанных данных (./train/prepared.csv)'
    assert 'prepared.csv' in os.listdir('./test'), 'Нет предобработанных данных (./test/prepared.csv)'

    model = LinearRegression()

    train_df = pd.read_csv(os.path.join('./train', 'prepared.csv'))

    model.fit(train_df.drop(columns=['label']), train_df['label'])

    pickle.dump(model, open('./fitted_model.bin', 'wb'))
