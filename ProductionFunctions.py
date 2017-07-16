import pandas as pd
import numpy as np

import theano
from theano import function
from theano import pp
from theano import tensor as T

class CobbDouglas:
	def __init__(self, data, labels):
		self.DATA = np.log(data)
		self.LABELS = labels
	
		self.learning_rate = 0.1
		self.n_dims = self.DATA.shape[1]
		self.n_steps = 20
		
	def findRegressionCoefficients(self):

		self.x = T.matrix(name='x')  #Input matrix with examples.
		self.w = theano.shared(value = np.zeros((self.n_dims, 1), dtype=theano.config.floatX), name='w', borrow=True)
		self.f = function([self.x], T.dot(self.w, self.x)) # Linear regression.
		
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
