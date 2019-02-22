import pickle
import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import keras.preprocessing.text
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import EarlyStopping

class RnaModule(object):
	# X for samples and Y for labels
	trainingX = []
	trainingY = []
	testX = []
	testY = []

	number_neurons_input_layer = 0
	number_neurons_hidden_layer = 0
	number_neurons_output_layer = 0

	input_dim_neurons = 0

	#funcoes de ativacao dos neuronios de cada camada
	activation_function_input_layer = "relu"
	activation_function_hidden_layer = "relu"
	activation_function_output_layer = "sigmoid"

	model = None

	classFeatureName = 'Label'

	def __init__(self):
		# print("ANN module constructor")
		pass

	#funcao para criar a rna para abordagem simples
	def generateModel(self):
		self.model = Sequential()
		self.model.add(Dense(self.number_neurons_input_layer, input_dim= self.input_dim_neurons, init='normal', activation=self.activation_function_input_layer))
		self.model.add(Dense(self.number_neurons_hidden_layer, init='normal', activation=self.activation_function_hidden_layer))
		self.model.add(Dense(self.number_neurons_output_layer, init='normal', activation=self.activation_function_output_layer))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		# passar por callback na funcao fit
		# csv_logger = CSVLogger('training.log')

		#funcao para interromper treinamento quando o erro for suficientemente pequeno
		early_stopping = EarlyStopping(monitor='loss',patience=20)
<<<<<<< HEAD
		history = self.model.fit(self.trainingX, self.trainingY, epochs=500, verbose=1, callbacks=[early_stopping])
=======
		fit = self.model.fit(self.data_set_samples, self.data_set_labels, epochs=500, verbose=2, callbacks=[early_stopping])
>>>>>>> 2da67dff10b351ca93c72e138f2b83474fbc06d4

    #funcao para criar a rna para a abordagem hibrida
	def generateHybridModel(self):
		self.model = Sequential()
		self.model.add(Dense(self.number_neurons_input_layer, input_dim= self.input_dim_neurons, init='normal', activation=self.activation_function_input_layer))
		self.model.add(Dense(self.number_neurons_hidden_layer, init='normal', activation=self.activation_function_hidden_layer))
		self.model.add(Dense(self.number_neurons_output_layer, init='normal', activation=self.activation_function_output_layer))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		csv_logger = CSVLogger('training.log')
		#funcao para interromper treinamento quando o erro for suficientemente pequeno
		early_stopping = EarlyStopping(monitor='loss', patience=20)

		history = self.model.fit(self.trainingX, self.trainingY, nb_epoch=500, verbose=2, callbacks=[early_stopping])

		#obter valores da camada de saida da ultima iteracao do treinamento
		get_3rd_layer_output = K.function([self.model.layers[0].input], [self.model.layers[2].output])
		layer_output = get_3rd_layer_output([self.trainingX])[0]

		predictions = self.model.predict_classes(self.trainingX)

		return layer_output, predictions, history

	#funcao utilizada para retornar o resultado da classificacao em termos de -1 a 1 (utilizada para a abordagem hibrida)
	def predict(self):
		predictions = self.model.predict(self.testX)
		return predictions

	#funcao utilizada para retornar o resultado da classificacao em 1 ou 0(utilizada para a abordagem simples)
	def predictClasses(self):
		predictions = self.model.predict_classes(self.testX)
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

	def setNumberNeuronsInputLayer(self, number):
		self.number_neurons_input_layer = number

	def getNumberNeuronsInputLayer(self):
		return self.number_neurons_input_layer

	def setNumberNeuronsHiddenLayer(self, number):
		self.number_neurons_hidden_layer = number

	def getNumberNeuronsHiddenLayer(self):
		return self.number_neurons_hidden_layer

	def setNumberNeuronsOutputLayer(self, number):
		self.number_neurons_output_layer = number

	def getNumberNeuronsOutputLayer(self):
		return self.number_neurons_output_layer

	def setActivationFunctionInputLayer(self, activation_function):
		self.activation_function_input_layer = activation_function

	def getActivationFunctionInputLayer(self):
		return self.activation_function_input_layer

	def setActivationFunctionHiddenLayer(self, activation_function):
		self.activation_function_hidden_layer = activation_function

	def getActivationFunctionHiddenLayer(self):
		return self.activation_function_hidden_layer

	def setActivationFunctionOutputLayer(self, activation_function):
		self.activation_function_output_layer = activation_function

	def getActivationFunctionOutputLayer(self):
		return self.activation_function_output_layer

	def setInputDimNeurons(self, number):
		self.input_dim_neurons = number

	def getNumberNeuronsInputLayer(self):
		return self.input_dim_neurons

	def setDimInputLayer(self, dim_input_layer):
		self.dim_input_layer = dim_input_layer

	def getDimInputLayer(self):
		return self.dim_input_layer

	def getModel(self):
		return self.model
