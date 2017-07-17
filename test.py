import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import ProductionFunctions as pf
from sklearn import neighbors

def findAccuracy(predicted, original):	#make this single line, if needed, or extend it to calculate other metrics
	N = len(predicted)
	accuracy = 0.0
	for i in range(N):
		if predicted[i] == original[i]:
			accuracy += 1
	return accuracy/N
		
		
data_file = 'data/iris.csv'
data_frame = pd.read_csv(data_file, sep = ',', header=0)
data_frame['l'] = data_frame['l'].map({'Iris-setosa':2, 'Iris-versicolor':3, 'Iris-virginica':4})


#DATASET = data_frame.values[:, 0:-1]
DATASET = data_frame.values[:, :4]
print(DATASET)
LABELS = data_frame.values[:, 4]
############################################################################################

results_original = []
results_processed = []

for iteration in range(0, 10):
	print('ITERATION = ', iteration+1)
	
	training_data, training_labels, test_data, test_labels = pf.randomSample(DATASET, LABELS, 0.8)
	prep = pf.CobbDouglas(training_data, training_labels)
	#print('Printing labels: ', training_labels)
	ELASTICITIES, CONSTANT = prep.findRegressionCoefficients()

	"""PROCESSED_TRAINING_DATA = pf.elasticExponentiation(training_data, ELASTICITIES, CONSTANT)
	PROCESSED_TEST_DATA = pf.elasticExponentiation(test_data, ELASTICITIES, CONSTANT)

	#Model implementation on original data
	clf = neighbors.KNeighborsClassifier(7)
	clf.fit(training_data, training_labels)
	predicted = clf.predict(test_data)
	results_original.append(findAccuracy(predicted, test_labels))
    
	#Model implementation on processed data
	clf_processed = neighbors.KNeighborsClassifier(7)
	clf_processed.fit(PROCESSED_TRAINING_DATA, training_labels)
	predicted_processed = clf_processed.predict(PROCESSED_TEST_DATA)
	results_processed.append(findAccuracy(predicted_processed, test_labels))

print('Results on original data: ', sum(results_original)/len(results_original))
print('Results on processed data: ', sum(results_processed)/len(results_processed))"""
