import numpy as np
import pickle
import os
from sklearn import metrics

if __name__ == '__main__':
    model = pickle.load(open('./fitted_model.bin', 'rb'))

    test_x = np.load(os.path.join('./data', 'test_x.npy'))
    test_y = np.load(os.path.join('./data', 'test_y.npy'))

    preds = model.predict(test_x)

    print(f'Model test accuracy is: {metrics.accuracy_score(test_y, preds):.3f}')
