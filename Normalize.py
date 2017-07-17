import pandas as pd
import numpy as np

class Normalize:
	def __init__(self, data, portion = 'test', range_min = 1.0, range_max = 10.0):

		if portion == 'train':
			self.minmax_list = []
			self.data_size = data.shape
			self.MIN = range_min
			self.MAX = range_max
			self.findMinMax(data)
			#print(self.minmax_list)
			#print(self.MIN, self.MAX)
		self.TRAINING_DATA = self.normalizeData(data)
		#print(self.minmax_list)
		#print(self.MIN, self.MAX)
		
	def normalizeData(self, data):
		column_index = 0
		for column_index in range(self.data_size[1]):
			#print(column_index, self.MAX, self.MIN)
			col_min, col_max = self.minmax_list[column_index]
			
			data[:,column_index] = (data[:,column_index] - col_min)/(col_max - col_min)
			data[:,column_index] = (data[:,column_index] * (self.MAX - self.MIN)) + self.MIN
			#print('Printing from here: ', data[:,column_index])

		return data
		
	def findMinMax(self, data):
		for column_index in range(data.shape[1]):
			column = data[:,column_index]
			col_min = int(min(column))
			col_max = int(max(column))
			self.minmax_list.append((col_min, col_max))			
	
	def getTrainData(self):
		return self.TRAINING_DATA
		
	def getTestData(self, data):
		#self.TEST_DATA = self.normalizeData(data)
		return self.normalizeData(data)
