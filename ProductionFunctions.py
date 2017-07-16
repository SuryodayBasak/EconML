import pandas as pd
import numpy as np

from random import random as rand
import theano
from theano import function
from theano import pp
from theano import tensor as T

class CobbDouglas:
	def __init__(self, data, labels):
		self.DATA = np.log(data)
		print(self.DATA)
		self.LABELS = labels
	
		self.learning_rate = 0.1
		self.n_dims = self.DATA.shape[1]
		self.n_steps = 20
		
	def findRegressionCoefficients(self):

		self.x = T.matrix(name='x')  #Input matrix with examples.
		self.c = theano.shared(value = 0.0, dtype=theano.config.floatX), name='w', borrow=True)  #Constant of regression
		self.w = theano.shared(value = np.zeros((self.n_dims, 1), dtype=theano.config.floatX), name='w', borrow=True)
		self.f = function([self.x], T.dot(self.w, self.x)) # Linear regression.
		#self.f = function([self.x], T.dot(self.w, self.x) + self.c) # Linear regression.
		
		self.y = T.vector(name='y')  # Output vector with y values.
		self.loss = T.mean((T.dot(self.x, self.w).T - self.y) ** 2) # Define loss function.
		self.g_loss = T.grad(self.loss, wrt=self.w) # Build the gradient descent algorithm.
		
		train_model = function(inputs=[], 
						outputs = self.loss, 
						updates=[(self.w, self.w - self.learning_rate * self.g_loss)], 
						givens={self.x: self.DATA, self.y: self.LABELS
						})

		for i in range(self.n_steps):
			#print ("cost", train_model())
			train_model()
		
		#print(self.w.eval())
		return np.squeeze(np.asarray(self.w.eval()))

def randomSample(DATASET, LABELS, ratio):
	if np.shape(DATASET)[0] != np.shape(LABELS)[0]:
		print("The size of the feature vectors and target variables are different.")
	
	training_data = []
	training_labels = []
	test_data = []
	test_labels = []
	
	for i in range(len(DATASET)):
		r = rand()
		if r < ratio:
			training_data.append(DATASET[i])
			training_labels.append(LABELS[i])
		else:
			test_data.append(DATASET[i])
			test_labels.append(LABELS[i])
	
	return np.matrix(training_data), np.array(training_labels), np.matrix(test_data), np.array(test_labels)

#WRITE THIS IN THEANO LATER
def elasticExponentiation(DATASET, ELASTICITIES):
	if np.shape(DATASET)[1] != np.shape(ELASTICITIES)[0]:
		print("Lengths of feature vectors and easticity vector are not the same")
	
	rows, cols = np.shape(DATASET)
	results = []
	for i in range(rows):
		answer = 1
		for j in range(cols):
			answer*= DATASET[i,j]**ELASTICITIES[j]
		results.append([answer])
	
	return results
	
