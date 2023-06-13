import pandas as pd
import pickle
import os
from sklearn import metrics

if __name__ == '__main__':
    assert 'prepared.csv' in os.listdir('./train'), 'Нет предобработанных данных (./train/prepared.csv)'
    assert 'prepared.csv' in os.listdir('./test'), 'Нет предобработанных данных (./test/prepared.csv)'
    assert 'fitted_model.bin' in os.listdir('./'), "Нет обученной модели (fitted_model.bin)"

    model = pickle.load(open('./fitted_model.bin', 'rb'))

    test_df = pd.read_csv(os.path.join('./test', 'prepared.csv'))

    preds = model.predict(test_df.drop(columns=['label']))

    print(f'Model test R2 is: {metrics.r2_score(test_df["label"].values, preds):.3f}')
