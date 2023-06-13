import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = []
    for harmonic in range(1, 20_000):
        data.append(np.hstack([np.sin(np.arange(200) * (harmonic % 200) + np.random.randn(200)), np.array([harmonic])]))

    pd_data = pd.DataFrame(data, columns=[f'feature_{x}' for x in range(200)] + ['label'])

    if 'train' not in os.listdir('./'):
        os.mkdir('train')

    if 'test' not in os.listdir('./'):
        os.mkdir('test')

    for idx in range(0, 20):
        if idx > 9:
            pd_data.iloc[idx * 1000: (idx + 1) * 1000].to_csv(
                os.path.join('test', f'data_chunk_{idx - 10}.csv'), index=False
            )
        else:
            pd_data.iloc[idx * 1000: (idx + 1) * 1000].to_csv(
                os.path.join('train', f'data_chunk_{idx}.csv'), index=False
            )
