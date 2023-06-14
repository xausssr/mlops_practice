import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    model = RandomForestClassifier(max_depth=4)

    train_x = np.load(os.path.join('./data', 'train_x.npy'))
    train_y = np.load(os.path.join('./data', 'train_y.npy'))

    model.fit(train_x, train_y)

    pickle.dump(model, open('./fitted_model.bin', 'wb'))
