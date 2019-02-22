import numpy as np
import sys, os
from dataSet import DataSet
import pandas as pd
import time
import glob
import pickle
import datetime
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/hybrid")
from hybrid_classifier import HybridClassifier
from evaluate_module import EvaluateModule
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/rna")
from rna_classifier import RnaClassifier
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/knn")
from knn_classifier import KnnClassifier
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/clusteredKnn")
from clustered_knn_classifier import ClusteredKnnClassifier
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/clusteredDensityKnn")
from clustered_density_knn_classifier import ClusteredDensityKnnClassifier


class CrossValidation(object):
	dts = None
	#metodo utilizado para classifacao
	classifier = None

	#conjunto de dados de teste
	testData = None
	#conjunto de dados de treinamento
	trainingData = None

	evaluate = None

	#numero de folds
	numberOfFolds = 10

	file_path = ""

	#caminho da pasta onde serao salvos os resultados
	result_path = ""
	preprocessor = None

	def __init__(self):
		# print("Cross Validation constructor")
		self.evaluate = EvaluateModule()

	def run(self):
		self.classifier.setResultPath(self.result_path)
		self.foldExecution()

	def foldExecution(self):
		i = self.iteration

		for self.iteration in range(i,(self.numberOfFolds+1)):
			tempo_inicio = time.time()
			self.loadTrainingData()
			self.loadTestData()

			#executa funcoes para transformacao de dados categoricos
			if self.preprocessor:
				self.preprocessor.setDataSet(self.trainingData)
				self.preprocessor.setTestDataSet(self.testData)

				self.trainingData, self.testData = self.preprocessor.transformCategory()

			#seta dados de treinamento e teste no classificador
			self.classifier.setDataSet(self.trainingData)
			self.classifier.setTestDataSet(self.testData)

			#seta iteracao do cross no classficador
			self.classifier.setIteration(self.iteration)
			#executa o processo de treino e teste do classificador
			self.classifier.run()

			del(self.trainingData)
			# self.loadTestData()
			#seta conjunto de dados original de teste e iteracao atual do cross-validation na classe de avaliacao
			self.evaluate.setTestDataSet(self.testData)
			self.evaluate.setIteration(self.iteration)

			#verifica quel o metodo de classificacao utilziado
			if(isinstance(self.classifier, RnaClassifier)):
				print("rna")
				self.evaluate.setResultPath( self.result_path)
			elif(isinstance(self.classifier, KnnClassifier)):
				print("knn")
				# self.evaluate.setResultPath(self.result_path)
			elif(isinstance(self.classifier, ClusteredKnnClassifier)):
				print("clustered knn")
				#self.evaluate.setResultPath(self.result_path)
			elif(isinstance(self.classifier, ClusteredDensityKnnClassifier)):
				print("clustered density knn")
				#self.evaluate.setResultPath(self.result_path)
			elif(isinstance(self.classifier, HybridClassifier)):
				print("hybrid")
				annModel = self.classifier.getRna().getModel()
				self.saveModelToFile(annModel, 'ann')
				self.evaluate.setResultPath( self.result_path+"final_method_classification/")

			tempo_execucao = time.time() - tempo_inicio
			self.evaluate.setTempoExecucao(tempo_execucao)
			self.evaluate.setTrainingTime(self.classifier.getTrainingTime())
			self.evaluate.setTestTime(self.classifier.getTestTime())
			#executa metodo de avaliacao
			self.evaluate.run()

	#carrega conjunto de treinamento de acordo coma iteracao atual do cross valiadation
	def loadTrainingData(self):
		#exclude current cross validation iteration corresponding fold
		trainFolds = glob.glob(self.file_path + 'fold_[!' + str(self.iteration) + ']*.csv')
		self.trainingData = pd.concat((pd.read_csv(fold) for fold in trainFolds))

	#carrega conjunto de teste de acordo com a iteracao atual do cross validation
	def loadTestData(self):
		self.testData = DataSet.loadSubDataSet(self.file_path + "fold_" + str(self.iteration) + ".csv")

	def setIteration(self, iteration):
		self.iteration = iteration

	def setClassifier(self, classifier):
		self.classifier = classifier

	def getClassifier(self):
		return classifier

	def setPreprocessor(self, preprocessor):
		self.preprocessor = preprocessor

	def getPreprocessor(self):
		return preprocessor

	def setEvaluateModule(self, evaluate):
		self.evaluate = evaluate

	def getEvaluateModule(self):
		return evaluate

	def setFilePath(self, file_path):
		self.file_path = file_path

	def setResultPath(self, result_path):
		directory = os.path.dirname(result_path)
		if not os.path.exists(directory):
			os.makedirs(directory)

		self.result_path = result_path

	def setK(self, k):
		self.numberOfFolds = k

	def saveModelToFile(self, model, prefix):
		directory = os.path.dirname(self.file_path + 'models/')
		if not os.path.exists(directory):
			os.makedirs(directory)

		fileName = directory + prefix + '_' + str(self.iteration - 1)
		pickle.dump(model, open(fileName, 'wb'))
		print('[' + str(datetime.datetime.now()).split('.')[0] + '] ' + fileName + ' saved [')
