import pandas as pd
import numpy as np

data_file = 'iris.csv'
data_frame = pd.read_csv(data_file, sep = ',', header=0)
data_frame['l'] = data_frame['l'].map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})

DATASET = np.log(data_frame.values[:, 0:-1])
LABELS = data_frame.values[:, 4]

