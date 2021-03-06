import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import ProductionFunctions as pf
import Normalize as norm
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model

def findAccuracy(predicted, original):	#make this single line, if needed, or extend it to calculate other metrics
	N = len(predicted)
	accuracy = 0.0
	for i in range(N):
		if predicted[i] == original[i]:
			accuracy += 1
	return accuracy/N
		
		
data_file = 'data/iris.csv'
#data_file = 'data/skin_seg.csv'
data_frame = pd.read_csv(data_file, sep = ',', header=0)
data_frame['l'] = data_frame['l'].map({'Iris-setosa':2, 'Iris-versicolor':3, 'Iris-virginica':4})


#DATASET = data_frame.values[:, 0:-1]
DATASET = data_frame.values[:, :4]
#print(DATASET)
LABELS = data_frame.values[:, 4]
############################################################################################

results_original = []
results_processed = []

for iteration in range(0, 10):
	print('ITERATION = ', iteration+1)
	
	training_data, training_labels, test_data, test_labels = pf.randomSample(DATASET, LABELS, 0.8)
	training_data_bk = training_data.copy()
	test_data_bk = test_data.copy()
	training_labels_bk = training_labels.copy()
	test_labels_bk = test_labels.copy()
	
	#norm_obj = norm.Normalize(training_data, test_data)
	norm_obj = norm.Normalize(training_data, 'train')
	norm_training_data = norm_obj.getTrainData()
	#print(norm_training_data)
	norm_test_data = norm_obj.getTestData(test_data)
	#print(norm_test_data)
	
	#prep = pf.CobbDouglas(training_data, training_labels)
	#print('Printing labels: ', training_labels)
	#ELASTICITIES, CONSTANT = prep.findRegressionCoefficients()
	
	prep = pf.CobbDouglas(norm_training_data, training_labels)
	#print('Printing labels: ', training_labels)
	ELASTICITIES, CONSTANT = prep.findRegressionCoefficients()

	PROCESSED_TRAINING_DATA = pf.elasticExponentiation(norm_training_data, ELASTICITIES, CONSTANT)
	PROCESSED_TEST_DATA = pf.elasticExponentiation(norm_test_data, ELASTICITIES, CONSTANT)
	#print(PROCESSED_TEST_DATA)
	
	#Model implementation on original data
	#clf = neighbors.KNeighborsClassifier(7)
	clf = svm.SVC()
	print(training_data_bk)
	clf.fit(training_data_bk, training_labels_bk)
	predicted = clf.predict(test_data_bk)
	results_original.append(findAccuracy(predicted, test_labels_bk))
    
	#Model implementation on processed data
	#clf_processed = neighbors.KNeighborsClassifier(7)
	#clf_processed = svm.SVC()
	#clf_processed.fit(PROCESSED_TRAINING_DATA, training_labels)
	
	clf_processed = linear_model.LogisticRegression(C=1e5)
	clf_processed.fit(PROCESSED_TRAINING_DATA, training_labels)
	predicted_processed = clf_processed.predict(PROCESSED_TEST_DATA)
	results_processed.append(findAccuracy(predicted_processed, test_labels))

print('Results on original data: ', sum(results_original)/len(results_original))
print('Results on processed data: ', sum(results_processed)/len(results_processed))
