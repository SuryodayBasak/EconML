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
		self.DATA = np.c_[self.DATA, np.ones(np.shape(self.DATA)[0])]
		#column_ones = np.ones(np.shape(self.DATA)[0])
		#self.DATA = np.append(self.DATA, column_ones, 1)
		print(self.DATA)
		self.LABELS = labels
	
		self.learning_rate = 0.001
		self.n_samples, self.n_dims = self.DATA.shape
		self.n_steps = 10000
		
	def findRegressionCoefficients(self):

		self.x = T.matrix(name='x')  #Input matrix with examples.
		self.y = T.vector(name='y')  # Output vector with y values.
		self.w = theano.shared(value = np.zeros((self.n_dims, 1), dtype=theano.config.floatX), name='w', borrow=True)  #Constant of regression


		"""
		self.cost = T.sum(T.pow(self.prediction-self.y,2))/(2*self.n_samples)
		self.gradw = T.grad(self.cost, self.w)
		self.gradc = T.grad(self.cost, self.c)

		train_model = function(inputs=[], 
						outputs = self.cost, 
						updates=[(self.w, self.w - self.learning_rate * self.gradw), (self.c, self.c - self.learning_rate * self.gradc)], 
						givens={self.x: self.DATA, self.y: self.LABELS
						})
		
		for i in range(self.n_steps):
			print ("costi", train_model())
		#self.train_model = theano.function([self.x, self.y], self.cost, self.updates = [(self.w, self.w - self.lr * self.gradw), (self.c, self.c - self.lr * self.gc)])
		
		self.train_model = theano.function(inputs = [],
												outputs = self.cost,
												updates = [(self.w, self.w - self.learning_rate * self.gradw),
												givens = {self.x: self.DATA, self.y: self.LABELS
												})
									
		for i in range(self.n_steps):
			print ("costi", self.train_model())
		"""
			
		#self.f = function([self.x], T.dot(self.w, self.x)) # Linear regression.
		self.f = function([self.x], T.dot(self.w, self.x)) # Linear regression.
		
		self.loss = T.mean((T.dot(self.x, self.w).T - self.y) ** 2) # Define loss function.
		self.g_loss = T.grad(self.loss, wrt = self.w) # Build the gradient descent algorithm.
		
		train_model = function(inputs=[], 
						outputs = self.loss, 
						updates=[(self.w, self.w - self.learning_rate * self.g_loss)], 
						givens={self.x: self.DATA, self.y: self.LABELS
						})
		#print('reached here')
		for i in range(self.n_steps):
			#print ("cost "+str(i+1), train_model())
			train_model()

		
		self.WEIGHTS = np.squeeze(np.asarray(self.w.eval()))
		print(self.WEIGHTS)
		self.ELASTICITIES = self.WEIGHTS[:-1]
		self.CONSTANT = self.WEIGHTS[self.n_dims-1]
		return self.ELASTICITIES, self.CONSTANT

#REDO THIS IN LESSER LINES OF CODE
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
def elasticExponentiation(DATASET, ELASTICITIES, CONSTANT):
	if np.shape(DATASET)[1] != np.shape(ELASTICITIES)[0]:
		print("Lengths of feature vectors and easticity vector are not the same")
	
	rows, cols = np.shape(DATASET)
	results = []
	for i in range(rows):
		answer = np.e**CONSTANT
		for j in range(cols):
			answer*= DATASET[i,j]**ELASTICITIES[j]
		results.append([answer])
		#results.append(answer)
	
	return results
	
