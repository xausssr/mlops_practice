

import pickle
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

    
def test_model(dataset):
    score = pickle_model.score(X_test, y_test).round(4)
    assert score >= 0.70, 'Corrupted data set: '+ dataset
    
    
with open('model.pkl', 'rb') as f: 
    pickle_model = pickle.load(f)

    
for dataset in os.listdir('datasets'):
    with open('./datasets/' + dataset, 'rb') as f:
        data = np.load(f)
        X_train, X_test, y_train, y_test = train_test_split(*data)
        test_model(dataset)
