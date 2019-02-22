import numpy as np
import pandas
import os
from sklearn import neighbors

class KnnModule(object):
	#conjuto de exemplos de treino
	trainingX = []
	#classes dos exemplos de treino
	trainingY = []
	#conjunto de exemplos de teste
	testX = []
	#classes dos exemplos de teste
	testY = []
	k_neighbors = 1
	clf = None

	classFeatureName = 'Label'

	def __init__(self):
		# print("init knn module")
		pass

	#funcao que cria a base de exemplos do KNN
	def buildExamplesBase(self):
		self.clf = neighbors.KNeighborsClassifier(self.k_neighbors, weights='uniform', algorithm='brute')
		self.clf.fit(self.trainingX, self.trainingY)

	#funcao que realiza a classificacao dos exemplos
	def run(self):
		predictions = self.clf.predict(self.testX)
		return predictions

	def setDataSet(self, dataset):
		self.trainingX = dataset.drop([self.classFeatureName], axis = 1) #all instances with no class feature
		self.trainingY = getattr(dataset, self.classFeatureName).values #class feature of all instances
		#print(self.trainingX)
		#print(self.trainingY)

	def setTestDataSet(self, dataset):
		self.testX = dataset.drop([self.classFeatureName], axis = 1) #all instances with no class feature
		self.testY = getattr(dataset, self.classFeatureName).values #class feature of all instances
		#print(self.testX)
		#print(self.testY)

	def setKNeighbors(self, k_neighbors):
		self.k_neighbors = k_neighbors

	def getKNeighbors(self):
		return self.k_neighbors
