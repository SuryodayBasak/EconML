import pandas as pd

data_file = 'iris.csv'
DATA = pd.read_csv(data_file, sep = ',', header=0)
DATA['l'] = DATA['l'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

